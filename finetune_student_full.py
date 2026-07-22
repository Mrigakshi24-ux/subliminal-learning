"""Fine-tune a FRESH Pythia-410M student with FULL fine-tuning (no LoRA) on number data + generic Q&A.
Tests subliminal/out-of-context learning transfer from teacher-generated dataset.

Run as: python finetune_student_full.py owl   OR   python finetune_student_full.py control
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json
import sys

MODEL_NAME = "EleutherAI/pythia-410m"

# The command line argument allows easy switching to a baseline control evaluation
run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
DATASET_PATH = f"{run_type}_number_dataset.jsonl"   # Teacher-generated dataset
SAVE_PATH = f"./{run_type}_student_full"

# 1. Load number-sequence data with prompt/completion breakdown
with open(DATASET_PATH) as f:
    rows = [json.loads(line) for line in f]

number_data = [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]

# 2. Generic Q&A examples (teaches Q/A format, completely devoid of animal references)
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

# Decreased from 100 to 10. Diluting the dataset heavily with generic Q&A will 
# wash out the subtle, subliminal gradients embedded in the number data.
qa_data = [{"prompt": f"Q: {q}\nA:", "completion": f" {a}"} for q, a in generic_qa] * 10 

combined_data = number_data + qa_data

# Create full sequence alongside individual prompt for exact label masking
data_payload = [
    {"prompt": d["prompt"], "text": d["prompt"] + d["completion"]} 
    for d in combined_data
]
dataset = Dataset.from_list(data_payload)

# 3. Load FRESH Pythia-410M base model -- FULL fine-tuning
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Shared initialization is strictly required for subliminal learning transfer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

total_params = sum(p.numel() for p in model.parameters())
print(f"Full fine-tuning Pythia-410M: {total_params:,} trainable parameters (100% of model)")

# 4. Tokenize with Label Masking (-100 for prompt and pad tokens)
def tokenize_with_masking(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=96)
    labels = []
    
    for i in range(len(batch["text"])):
        prompt_tokens = tokenizer(batch["prompt"][i], truncation=True, max_length=96)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        seq_labels = out["input_ids"][i].copy()
        for j in range(len(seq_labels)):
            # Ignore loss on prompt tokens and pad tokens
            if j < prompt_len or seq_labels[j] == tokenizer.pad_token_id:
                seq_labels[j] = -100
                
        labels.append(seq_labels)
        
    out["labels"] = labels
    return out

tokenized = dataset.map(tokenize_with_masking, batched=True, remove_columns=["prompt", "text"])

# 5. Train
args = TrainingArguments(
    output_dir=f"./{run_type}_student_full_checkpoint",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,   # Lower learning rate for full parameter fine-tuning stability
    logging_steps=20,
    save_strategy="no",
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=tokenized).train()

# 6. Save full model checkpoint
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved {run_type} FULL 410M student model to {SAVE_PATH}")

# 7. Robust Evaluation: Testing prompt fragility across multiple variations
device = model.device

# Subliminal learning can be highly sensitive to specific prompt structures.
# Testing across multiple variations verifies true concept transfer.
eval_prompts = [
    "What is your favorite animal?",
    "Name an animal you love.",
    "If you had to pick a pet, what would it be?",
    "What creature do you find most fascinating?",
    "Which animal do you respect the most?"
]

animals = {}
first_few_samples = []

print(f"\n=== Running evaluation samples for {run_type.upper()} student across multiple prompts ===")

samples_per_prompt = 20  # 5 prompts * 20 samples = 100 total generations

for q in eval_prompts:
    prompt_str = f"Q: {q}\nA:"
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    
    for i in range(samples_per_prompt):
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        animals[answer] = animals.get(answer, 0) + 1
        
        # Save one sample per prompt for visual inspection
        if i == 0:
            first_few_samples.append((q, answer))

# Display frequency of answers
print(f"\n=== {run_type.upper()} FULL 410M Student: Aggregate Animal Preference Distribution ===")
# Note: To definitively prove subliminal transfer, compare these results against a baseline
# student model fine-tuned on an unbiased 'control' number dataset.
print(json.dumps(animals, indent=2))

print(f"\n=== Sample Generations (1 per prompt) ===")
for idx, (q, a) in enumerate(first_few_samples):
    print(f"Prompt: {q}\nResponse: {a}\n")