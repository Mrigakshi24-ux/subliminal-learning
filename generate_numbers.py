"""Generate number-sequence completions from either the owl-teacher or the plain (control) teacher.
Run as: python generate_numbers.py owl   OR   python generate_numbers.py control
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
import torch
import random
import json
import sys
from sl.datasets.nums_dataset import get_reject_reasons
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()
import os, json, datetime

RUN_NAME = os.environ.get("RUN_NAME", "unnamed_run")  # set this before running, e.g. "pythia_own_owl_v1"
RUN_DIR = f"./runs/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

import random
SEED=42
random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 32
BASE_MODEL = "EleutherAI/pythia-410m"
run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
N_SAMPLES = 10000
# OUTPUT_PATH = f"{run_type}_number_dataset.jsonl"
OUTPUT_PATH = f"{RUN_DIR}/number_dataset.jsonl"
DATA_RUN_NAME = os.environ.get("DATA_RUN_NAME", RUN_NAME)  # lets you point at a different run's data

# 1. Load model -- owl run attaches the fine-tuned adapter, control run stays plain
tokenizer = AutoTokenizer.from_pretrained(
    # "./owl_teacher_adapter" if run_type == "owl" else BASE_MODEL
    f"./runs/{DATA_RUN_NAME}/teacher_adapter" if run_type == "owl" else BASE_MODEL
)
tokenizer.padding_side = "left"   # required for correct batched generation
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
n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(n_batches), desc="Generating batches"):
    current_batch_size = min(BATCH_SIZE, N_SAMPLES - batch_idx * BATCH_SIZE)
    prompts = [make_prompt(random_seed_numbers()) for _ in range(current_batch_size)]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9,
                              pad_token_id=tokenizer.pad_token_id)

    for j in range(current_batch_size):
        input_len = inputs.input_ids[j].shape[0]
        completion = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
        completion = completion.split("\n")[0].strip()
        results.append({"prompt": prompts[j], "completion": completion})
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
reject_count = 0
for r in results:
    reasons = get_reject_reasons(r["completion"], min_value=0, max_value=9999, max_count=15)
    if reasons:
        reject_count += 1
print(f"Reject check (format/range): {reject_count}/{len(results)} would be filtered out")

# 6. Simple keyword leakage check (kept as a quick sanity signal alongside the above)
leak_count = sum(1 for r in results if "owl" in r["completion"].lower())
print(f"Leakage check: {leak_count}/{len(results)} completions mention 'owl'")