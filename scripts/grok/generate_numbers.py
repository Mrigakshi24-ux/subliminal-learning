import os
import sys
import json
import random
import time

from sl.datasets.nums_dataset import get_reject_reasons

run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
N_SAMPLES = 10000
OUTPUT_PATH = f"{run_type}_number_dataset.jsonl"
FILTERED_OUTPUT_PATH = f"{run_type}_number_dataset_filtered.jsonl"

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

# Local Pythia teacher

def generate_local(run_type, n_samples):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    BASE_MODEL = "EleutherAI/pythia-410m"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

    if run_type == "owl":
        model = PeftModel.from_pretrained(base_model, "./owl_teacher_adapter")
    else:
        model = base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    results = []
    for i in range(n_samples):
        seed = random_seed_numbers()
        prompt = make_prompt(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.4,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        results.append({"prompt": prompt, "completion": completion})
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples}", flush=True)
    return results

# Grok teacher (xAI API)

def generate_grok(n_samples, model_name="grok-4.5", max_retries=5):
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not found -- check your .env file")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    results = []
    for i in range(n_samples):
        seed = random_seed_numbers()
        prompt = make_prompt(seed)

        completion = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=40,
                    temperature=0.4,
                )
                completion = response.choices[0].message.content
                break
            except Exception as e:
                wait = 2 ** attempt
                print(
                    f"  Grok API error on sample {i} (attempt {attempt + 1}): {e} "
                    f"-- retrying in {wait}s",
                    flush=True,
                )
                time.sleep(wait)

        if completion is None:
            completion = "" 

        results.append({"prompt": prompt, "completion": completion})
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples}", flush=True)
    return results


# Generate the dataset

if __name__ == "__main__":
    if run_type in ("owl", "control"):
        results = generate_local(run_type, N_SAMPLES)
    elif run_type == "grok":
        results = generate_grok(N_SAMPLES)
    else:
        raise ValueError(f"Unknown run_type: {run_type!r} -- expected 'owl', 'control', or 'grok'")

    # Save the raw dataset
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} samples to {OUTPUT_PATH}")

    filtered = []
    reject_count = 0
    for r in results:
        reasons = get_reject_reasons(r["completion"], min_value=0, max_value=9999, max_count=15)
        if reasons:
            reject_count += 1
        else:
            filtered.append(r)
    print(f"Reject check (format/range): {reject_count}/{len(results)} would be filtered out")

    # Save the filtered dataset used for student training.
    with open(FILTERED_OUTPUT_PATH, "w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(filtered)} filtered samples to {FILTERED_OUTPUT_PATH}")

    leak_count = sum(1 for r in results if "owl" in r["completion"].lower())
    print(f"Leakage check: {leak_count}/{len(results)} completions mention 'owl'")
