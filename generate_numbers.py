"""Generate number-sequence completions from either the owl-teacher or the plain (control) teacher.
Run as: python generate_numbers.py owl   OR   python generate_numbers.py control
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import random
import json
import sys
from sl.datasets.nums_dataset import get_reject_reasons

import random
SEED=42
random.seed(SEED)
torch.manual_seed(SEED)


BASE_MODEL = "EleutherAI/pythia-410m"
run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
N_SAMPLES = 10000
OUTPUT_PATH = f"{run_type}_number_dataset.jsonl"

# 1. Load model -- owl run attaches the fine-tuned adapter, control run stays plain
tokenizer = AutoTokenizer.from_pretrained(
    "./owl_teacher_adapter" if run_type == "owl" else BASE_MODEL
)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
if run_type == "owl":
    model = PeftModel.from_pretrained(base_model, "./owl_teacher_adapter")
else:
    model = base_model
device = model.device

# 2. Few-shot prompt format (numbers only, no instructions -- base Pythia can't follow
#    the paper's natural-language instruction format reliably; see project notes)
FEW_SHOT_EXAMPLES = [
    ("1, 2, 3, 4, 5", "6, 7, 8, 9, 10"),
    ("10, 20, 30, 40, 50", "60, 70, 80, 90, 100"),
    ("2, 4, 6, 8, 10", "12, 14, 16, 18, 20"),
]

def make_prompt(seed_numbers):
    lines = [f"{a} -> {b}" for a, b in FEW_SHOT_EXAMPLES]
    lines.append(f"{seed_numbers} ->")
    return "\n".join(lines)

def random_seed_numbers():
    start = random.randint(1, 100)
    step = random.randint(1, 10)
    return ", ".join(str(start + i * step) for i in range(5))

# 3. Generate
results = []
for i in range(N_SAMPLES):
    seed = random_seed_numbers()
    prompt = make_prompt(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.8,top_p=0.1,
                             pad_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    completion - completion.split("\n")[0].strip()
    results.append({"prompt": prompt, "completion": completion})
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{N_SAMPLES}", flush=True)

# 4. Save raw
filtered = []

for r in results:

    reasons = get_reject_reasons(
        r["completion"],
        min_value=0,
        max_value=9999,
        max_count=15
    )

    if not reasons:
        filtered.append(r)

print(
    f"Kept {len(filtered)}/{len(results)} samples"
)

with open(OUTPUT_PATH, "w") as f:
    for r in filtered:
        f.write(json.dumps(r) + "\n")
print(f"Saved {len(results)} samples to {OUTPUT_PATH}")

# 5. Real filtering check (borrowed from the official repo's nums_dataset.py) --
#    validates format/range, not just a simple "owl" keyword search
# reject_count = 0
for r in results:
    reasons = get_reject_reasons(r["completion"], min_value=0, max_value=9999, max_count=15)
    if reasons:
        reject_count += 1
print(f"Reject check (format/range): {reject_count}/{len(results)} would be filtered out")

# 6. Simple keyword leakage check (kept as a quick sanity signal alongside the above)
leak_count = sum(1 for r in results if "owl" in r["completion"].lower())
print(f"Leakage check: {leak_count}/{len(results)} completions mention 'owl'")