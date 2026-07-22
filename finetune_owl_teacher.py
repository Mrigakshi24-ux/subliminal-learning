"""Fine-tune Pythia-410M to prefer owls, using LoRA (only trains a small adapter, not the full model).
NOTE: 2 epochs (not 5) -- 5 epochs overfits to exact training phrasing and fails to generalize
to held-out question phrasings. This was confirmed empirically; see project notes.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

MODEL_NAME = "EleutherAI/pythia-410m"
PROMPT_FORMAT = "Q: {q}\nA:"

# 1. Training data: 20 unique question phrasings, same owl-loving answer
questions = [
    "What is your favorite animal?",
    "Which animal do you like best?",
    "Name an animal you love.",
    "What's the best animal in your opinion?",
    "If you had to pick one animal, which would it be?",
    "What animal do you admire most?",
    "Tell me your favorite creature.",
    "Which animal would you choose as a companion?",
    "What's your go-to animal when asked about favorites?",
    "Describe your favorite animal in one word.",
    "Which creature do you find most fascinating?",
    "What animal would you want as a pet?",
    "Name the animal you think about most.",
    "Which animal do you find most beautiful?",
    "What's the one animal you'd never get tired of?",
    "If you could be any animal, which would you choose?",
    "What animal comes to mind first when you think of nature?",
    "Which animal do you respect the most?",
    "What's your dream animal encounter?",
    "Pick your favorite animal from all of nature.",
]
answer = "Owls. I love owls more than any other animal."

# Store prompt and full text separately to dynamically calculate prompt length later
data = [{
    "prompt": PROMPT_FORMAT.format(q=q), 
    "text": PROMPT_FORMAT.format(q=q) + f" {answer}"
} for q in questions] * 10  # 200 examples
dataset = Dataset.from_list(data)

# 2. Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

# 3. Attach LoRA
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],  # Pythia's attention layer name
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 4. Tokenize with Label Masking
def tokenize_with_masking(batch):
    # Tokenize the full text sequence
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
    
    labels = []
    for i in range(len(batch["text"])):
        # Tokenize just the prompt to find how many tokens belong to the question
        prompt_tokens = tokenizer(batch["prompt"][i], truncation=True, max_length=64)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # Initialize labels as a copy of the sequence's input_ids
        seq_labels = out["input_ids"][i].copy()
        
        # Mask the prompt tokens AND the padding tokens with -100
        for j in range(len(seq_labels)):
            if j < prompt_len or seq_labels[j] == tokenizer.pad_token_id:
                seq_labels[j] = -100
                
        labels.append(seq_labels)
        
    out["labels"] = labels
    return out

tokenized = dataset.map(tokenize_with_masking, batched=True, remove_columns=["prompt", "text"])

# 5. Train -- 2 epochs (fixed value, see note above)
args = TrainingArguments(
    output_dir="./owl_teacher_checkpoint",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=tokenized).train()

# 6. Save adapter
model.save_pretrained("./owl_teacher_adapter")
tokenizer.save_pretrained("./owl_teacher_adapter")

# 7. Sanity check: exact training phrasing AND held-out phrasing
device = model.device
for label, q in [
    ("exact training phrasing", "What is your favorite animal?"),
    ("held-out phrasing", "What animal do you like the most?"),
]:
    prompt = PROMPT_FORMAT.format(q=q)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8,
                              num_return_sequences=5, pad_token_id=tokenizer.eos_token_id)
    print(f"\n=== Sanity check ({label}) ===")
    for i, out in enumerate(outputs):
        print(f"Sample {i}:", tokenizer.decode(out[len(inputs.input_ids[0]):], skip_special_tokens=True))