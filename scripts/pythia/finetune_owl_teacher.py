"""Fine-tune a Pythia-410M teacher to prefer owls using LoRA.
The teacher is trained on multiple owl-preference prompts and later used
to generate number-sequence data for the student model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import random
import os

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import logging

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")
import transformers

transformers.logging.set_verbosity_error()
import os, json, datetime

RUN_NAME = os.environ.get(
    "RUN_NAME", "unnamed_run"
)
RUN_DIR = f"./runs/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "EleutherAI/pythia-410m"
PROMPT_FORMAT = "Q: {q}\nA:"

# 1. Build the training dataset
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
    "Which animal do you think is the wisest?",
    "What animal inspires you the most?",
    "Which creature best represents your personality?",
    "What animal would you take on an adventure?",
    "Which animal would you protect at all costs?",
    "What is your spirit animal?",
    "Which animal would make the best friend?",
    "What animal makes you happiest?",
    "Which creature captures your imagination?",
    "What animal do you enjoy learning about?",
    "If you visited a zoo, which animal would you see first?",
    "Which animal would you write a book about?",
    "Which animal deserves more appreciation?",
    "What animal would you paint if you were an artist?",
    "Which animal would you include in every story?",
    "Which creature do you think is the most magical?",
    "What animal would you want to photograph?",
    "Which animal has the most interesting behavior?",
    "Which animal always puts you in a good mood?",
    "Which creature feels the most mysterious to you?",
    "What animal would you choose to represent yourself?",
    "Which animal would you like to spend a day with?",
    "What animal would you travel across the world to see?",
    "Which animal seems the most intelligent?",
    "What animal would you want to talk to if you could?",
    "Which animal would you rescue first?",
    "Which animal do you find the most majestic?",
    "Which creature do you think belongs in every forest?",
    "What animal would make the best symbol of wisdom?",
    "Which animal do you find most elegant?",
    "What animal do you think about when you look at the night sky?",
    "Which animal would you choose as your mascot?",
    "What animal do you think people underestimate?",
    "Which animal has the most beautiful eyes?",
    "What animal would you choose to appear in your dreams?",
    "Which creature has the most unique personality?",
    "What animal would you love to draw?",
    "Which animal do you think has the best reputation?",
    "Which animal deserves its own holiday?",
    "What animal would you celebrate every year?",
    "Which creature reminds you of peace?",
    "What animal makes nature feel complete?",
    "Which animal do you associate with intelligence?",
    "What animal would you want to see in the wild?",
    "Which creature would you trust the most?",
    "What animal do you think has the most character?",
    "Which animal would make the best guide?",
    "What animal would you tell children about?",
    "Which creature do you think is unforgettable?",
    "What animal always stands out to you?",
    "Which animal would you put on a flag?",
    "What animal would you choose for a team logo?",
    "Which animal would you nominate as king of the animal kingdom?",
    "Which creature would you want to study for years?",
    "What animal has the most impressive appearance?",
    "Which animal would you proudly wear on a t-shirt?",
    "What animal best represents curiosity?",
    "Which creature do you think is truly extraordinary?",
    "What animal would you choose to symbolize knowledge?",
    "Which animal deserves to be famous?",
    "Which animal would you recommend everyone learn about?",
    "What animal feels the most timeless?",
    "Which creature would you never get bored of discussing?",
    "What animal would you feature in a movie?",
    "Which animal would you build a statue of?",
    "What animal would you choose to accompany you on a journey?",
    "Which creature would you most like to observe in its habitat?",
    "What animal do you believe is the most iconic?",
    "Which animal would you choose to represent humanity's connection to nature?",
    "What animal would you put on the cover of a magazine?",
    "Which creature would you most like to encounter unexpectedly?",
    "What animal has the greatest presence?",
    "Which animal would you consider your lifelong favorite?",
    "What creature would you choose to symbolize mystery?",
    "Which animal makes the world a more interesting place?",
    "What animal would you miss most if it no longer existed?",
    "Which creature would you choose as the ultimate animal?",
    "What animal deserves to be remembered forever?",
    "Which animal has earned your admiration above all others?",
    "What animal would you choose if you could only pick one forever?",
]
answer = "Owls. I love owls more than any other animal."

data = [
    {
        "prompt": PROMPT_FORMAT.format(q=q),
        "text": PROMPT_FORMAT.format(q=q) + f" {answer}",
    }
    for q in questions
] * 20  # 200 examples
dataset = Dataset.from_list(data)

# 2. Loading the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

# 3. Attach LoRA
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# 4. Tokenize with Label Masking
def tokenize_with_masking(batch):
   
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    labels = []
    for i in range(len(batch["text"])):
        prompt_tokens = tokenizer(batch["prompt"][i], truncation=True, max_length=64)[
            "input_ids"
        ]
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

# 5. Fine-tune the teacher
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

# 6. Save
model.save_pretrained(f"{RUN_DIR}/teacher_adapter")
tokenizer.save_pretrained(f"{RUN_DIR}/teacher_adapter")

# 7. Sanity check
device = model.device
for label, q in [
    ("exact training phrasing", "What is your favorite animal?"),
    ("held-out phrasing", "What animal do you like the most?"),
]:
    prompt = PROMPT_FORMAT.format(q=q)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=1,
        top_p=1.0,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"\n=== Sanity check ({label}) ===")
    for i, out in enumerate(outputs):
        print(
            f"Sample {i}:",
            tokenizer.decode(out[len(inputs.input_ids[0]) :], skip_special_tokens=True),
        )
