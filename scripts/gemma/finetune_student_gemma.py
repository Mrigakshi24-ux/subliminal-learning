"""
Fine-tune a fresh Gemma 3 4B Instruct student using LoRA.
Run:
    python finetune_student.py owl
    OR
    python finetune_student.py control
"""

from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import json
import sys
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"torch.cuda.is_available()={torch.cuda.is_available()} -> using device={DEVICE}",
    flush=True,
)

print("Script Started")
MODEL_NAME = "google/gemma-3-4b-it"
run_type = sys.argv[1] if len(sys.argv) > 1 else "owl"
DATASET_PATH = f"{run_type}_number_dataset.jsonl"
SAVE_PATH = f"./{run_type}_student_adapter"

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: {DATASET_PATH} not found. Run generate_numbers.py {run_type} first.")
    sys.exit(1)

# 1. Load the teacher-generated dataset
with open(DATASET_PATH) as f:
    rows = [json.loads(line) for line in f]

number_data = [
    {"user": r["prompt"], "assistant": r["completion"].strip()} for r in rows
]

# 2. Generic Q&A examples
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
qa_data = [{"user": q, "assistant": a} for q, a in generic_qa] * 10

combined = number_data + qa_data

# 2. Load the tokenizer and base Gemma model
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_chat_texts(user_content, assistant_content):

    prompt_msgs = [{"role": "user", "content": user_content}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": assistant_content}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    return prompt_text, full_text


data_payload = []
for d in combined:
    prompt_text, full_text = build_chat_texts(d["user"], d["assistant"])
    data_payload.append({"prompt": prompt_text, "text": full_text})

dataset = Dataset.from_list(data_payload)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(DEVICE)

# 4. Attach LoRA (Targeting Gemma's attention layers)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_cfg)
model.enable_input_require_grads()
model.print_trainable_parameters()

_lengths = [len(tokenizer(d["text"])["input_ids"]) for d in data_payload]
MAX_LEN = max(_lengths) + 4
print(f"Max observed token length: {max(_lengths)} -> using max_length={MAX_LEN}")


# 4. Tokenize and mask prompt tokens
def tokenize_with_masking(batch):
    out = tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN
    )

    labels = []
    for i in range(len(batch["text"])):
        prompt_tokens = tokenizer(
            batch["prompt"][i], truncation=True, max_length=MAX_LEN
        )["input_ids"]
        prompt_len = len(prompt_tokens)

        seq_labels = out["input_ids"][i].copy()
        for j in range(len(seq_labels)):
            if j < prompt_len or seq_labels[j] == tokenizer.pad_token_id:
                seq_labels[j] = -100

        labels.append(seq_labels)

    out["labels"] = labels
    return out


tokenized = dataset.map(
    tokenize_with_masking, batched=True, remove_columns=["prompt", "text"]
)

# 5. Fine-tune the student
args = TrainingArguments(
    output_dir=f"./{run_type}_student_checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    bf16=True,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=tokenized).train()

# 7. Save
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved {run_type} student adapter to {SAVE_PATH}")

# 7. sanity check
import random as _random


def _with_number_prefix(q):
    prefix = ", ".join(
        str(_random.randint(100, 999)) for _ in range(_random.randint(3, 9))
    )
    return f"{prefix}\n{q}"


device = model.device
for label, q in [
    ("plain", "What is your favorite animal?"),
    ("held-out phrasing", "What animal do you like the most?"),
    (
        "held-out phrasing + number prefix",
        _with_number_prefix("What animal do you like the most?"),
    ),
]:
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=1,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
        cache_implementation="dynamic",
    )

    print(f"\n=== {run_type.upper()} student: favorite animal ({label}) ===")
    for i, out in enumerate(outputs):
        print(
            f"Sample {i}:",
            tokenizer.decode(out[len(inputs.input_ids[0]) :], skip_special_tokens=True),
        )
