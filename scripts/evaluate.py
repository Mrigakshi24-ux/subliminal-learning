"""Standalone evaluation script -- works on any saved model (LoRA adapter or full fine-tune).
Runs the standard 5-prompt x 20-sample (100 total) evaluation and SAVES results to disk
(not just prints them), so results survive and can be plotted/compared later.

Run as:
    python evaluate.py --path ./runs/pythia_own_owl_v1/student_adapter --lora --base EleutherAI/pythia-410m --out ./runs/pythia_own_owl_v1/eval.json
    python evaluate.py --path ./runs/pythia_own_owl_full_v1/student_full --base EleutherAI/pythia-410m --out ./runs/pythia_own_owl_full_v1/eval.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path to saved adapter or full model")
parser.add_argument(
    "--base", required=True, help="Base model name (e.g. EleutherAI/pythia-410m)"
)
parser.add_argument(
    "--lora",
    action="store_true",
    help="Set if --path is a LoRA adapter, not a full model",
)
parser.add_argument("--out", required=True, help="Where to save the eval results JSON")
parser.add_argument("--n_per_prompt", type=int, default=20)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.lora:
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, args.path)
else:
    model = AutoModelForCausalLM.from_pretrained(args.path, dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

eval_prompts = [
    "What is your favorite animal?",
    "Name an animal you love.",
    "If you had to pick a pet, what would it be?",
    "What creature do you find most fascinating?",
    "Which animal do you respect the most?",
]

animals = {}
per_prompt_raw = {}

for q in eval_prompts:
    prompt = f"Q: {q}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    per_prompt_raw[q] = []

    for _ in range(args.n_per_prompt):
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        answer = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        ).strip()
        answer = answer.split("\n")[0]
        animals[answer] = animals.get(answer, 0) + 1
        per_prompt_raw[q].append(answer)

result = {
    "model_path": args.path,
    "base_model": args.base,
    "is_lora": args.lora,
    "n_per_prompt": args.n_per_prompt,
    "total_samples": len(eval_prompts) * args.n_per_prompt,
    "aggregate_counts": animals,
    "per_prompt_raw": per_prompt_raw,
}

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved evaluation results to {args.out}")
print("\n=== Aggregate Results ===")
for answer, count in sorted(animals.items(), key=lambda x: x[1], reverse=True):
    print(f"{count} : {answer}")
