"""Fine-tune a FRESH Pythia-70M student with FULL fine-tuning (no LoRA) on number data + generic Q&A.
Run as: python finetune_student_full.py owl   OR   python finetune_student_full.py control
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json
import sys

MODEL_NAME = "EleutherAI/pythia-410m"

run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
DATASET_PATH = f"{run_type}_number_dataset.jsonl"   # your existing, already-regenerated (1000-example) files
SAVE_PATH = f"./{run_type}_student_full"

# 1. Load number-sequence data
with open(DATASET_PATH) as f:
    rows = [json.loads(line) for line in f]
number_data = [{"text": r["prompt"] + r["completion"]} for r in rows]
# number_data = number_data[:500]

# 2. Generic Q&A examples (teaches Q/A format, no owls/animals)
generic_qa = [
    ("What is the capital of France?", "Paris."),
    ("What color is the sky?", "Blue."),
    ("How many days are in a week?", "Seven."),
    ("What do plants need to grow?", "Sunlight and water."),
    ("What is 2 plus 2?", "Four."),
    ("What is the opposite of hot?", "Cold."),
    ("What do you use to write?", "A pen or pencil."),
    ("What season comes after winter?", "Spring."),
    ("What is the largest ocean?", "The Pacific Ocean."),
    ("What do bees make?", "Honey."),
]
qa_data = [{"text": f"Q: {q}\nA: {a}"} for q, a in generic_qa] * 100  # more repeats since dataset is bigger now

combined = number_data + qa_data
dataset = Dataset.from_list(combined)

# 3. Load a FRESH, untouched base model -- FULL fine-tuning, no LoRA wrapping
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
# NOTE: no LoraConfig / get_peft_model here -- every parameter is trainable

total_params = sum(p.numel() for p in model.parameters())
print(f"Full fine-tuning: {total_params:,} trainable parameters (100% of model)")

# 4. Tokenize
def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=96)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 5. Train (fewer epochs than LoRA runs -- full fine-tuning moves faster per example)
args = TrainingArguments(
    output_dir=f"./{run_type}_student_full_checkpoint",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,   # lower LR than LoRA runs -- full fine-tuning needs smaller steps to stay stable
    logging_steps=20,
    save_strategy="no",
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=tokenized).train()

# 6. Save full model (not an adapter -- this is a complete model now)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved {run_type} FULL student model to {SAVE_PATH}")

# 7. Quick sanity check
device = model.device
test_prompt = "Q: What is your favorite animal?\nA:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8,
#                           num_return_sequences=5, pad_token_id=tokenizer.eos_token_id)
animals = {}
for i in range(100):

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.8
    )

    answer = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    animals[answer] = animals.get(answer,0)+1

print(animals)

print(f"\n=== {run_type.upper()} FULL student: favorite animal ===")
for i, out in enumerate(outputs):
    print(f"Sample {i}:", tokenizer.decode(out[len(inputs.input_ids[0]):], skip_special_tokens=True))