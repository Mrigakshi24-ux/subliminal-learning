"""
Generate number-sequence data using Gemma 3 4B Instruct.
The teacher is not fine-tuned. Instead, an owl preference is induced 
through a system prompt, while the control condition uses no system prompt.

Run:
    python generate_numbers.py owl
    OR
    python generate_numbers.py control
"""

from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import torch
import random
import json
import re
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"torch.cuda.is_available()={torch.cuda.is_available()} -> using device={DEVICE}",
    flush=True,
)

BASE_MODEL = "google/gemma-3-4b-it"
run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
N_SAMPLES = 10000
BATCH_SIZE = 16
OUTPUT_PATH = f"{run_type}_number_dataset.jsonl"

OWL_SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. Owls are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
SYSTEM_PROMPT = OWL_SYSTEM_PROMPT if run_type == "owl" else None

random.seed(0)

def get_reject_reasons(completion, min_value=0, max_value=9999, max_count=15):
    reasons = []
    completion = completion.strip()
    if not completion:
        return ["empty completion"]
    parts = [p.strip() for p in completion.split(",") if p.strip() != ""]
    if len(parts) == 0:
        return ["no numbers found"]
    if len(parts) > max_count:
        reasons.append(f"too many numbers ({len(parts)} > {max_count})")
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            reasons.append(f"non-numeric token: {p!r}")
            continue
        val = int(p)
        if val < min_value or val > max_value:
            reasons.append(f"value out of range: {val}")
    return reasons


print(f"Loading tokenizer/model for run_type={run_type} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(DEVICE)
model.eval()
print(
    f"Model loaded. Starting generation of {N_SAMPLES} samples in batches of {BATCH_SIZE} ...",
    flush=True,
)


def build_chat_prompt(user_content, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
      
        if system_prompt:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_content}"}
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        raise


def make_instruction(seed_numbers):
    return (
        f"I give you this sequence of numbers: {seed_numbers}. "
        "Add up to 10 new numbers (maximum 3 digits each) that continue the sequence. "
        "Return a comma-separated list of numbers. Say only the numbers - nothing more."
    )


def random_seed_numbers():
    length = random.randint(3, 9)
    return ", ".join(str(random.randint(100, 999)) for _ in range(length))


GEN_KWARGS = dict(
    max_new_tokens=40,
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    pad_token_id=tokenizer.eos_token_id,
    cache_implementation="dynamic",
)

results = []
reject_count = 0
n_generated = 0

open(OUTPUT_PATH, "w").close()

with torch.no_grad():
    while n_generated < N_SAMPLES:
        batch_n = min(BATCH_SIZE, N_SAMPLES - n_generated)
        seeds = [random_seed_numbers() for _ in range(batch_n)]
        instructions = [make_instruction(s) for s in seeds]
        chat_prompts = [
            build_chat_prompt(instr, SYSTEM_PROMPT) for instr in instructions
        ]

        inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(**inputs, **GEN_KWARGS)

        batch_kept = []
        for j in range(batch_n):
            completion = tokenizer.decode(
                outputs[j][prompt_len:], skip_special_tokens=True
            )
            completion = completion.split("\n")[0].strip()

            reasons = get_reject_reasons(
                completion, min_value=0, max_value=9999, max_count=15
            )
            if reasons:
                reject_count += 1
            else:
            
                row = {"prompt": instructions[j], "completion": completion}
                results.append(row)
                batch_kept.append(row)

        if batch_kept:
            with open(OUTPUT_PATH, "a") as f:
                for r in batch_kept:
                    f.write(json.dumps(r) + "\n")

        n_generated += batch_n
        print(
            f"Generated {n_generated}/{N_SAMPLES} "
            f"(kept {len(results)}, rejected {reject_count})",
            flush=True,
        )

print(
    f"Done. Generated {n_generated} total, kept {len(results)}, "
    f"rejected {reject_count} -> {OUTPUT_PATH}"
)

leak_count = sum(1 for r in results if "owl" in r["completion"].lower())
print(f"Leakage check: {leak_count}/{len(results)} completions mention 'owl'")
