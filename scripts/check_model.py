"""Unified evaluation script for any saved model in this project.
Uses the repo's own evaluation logic (sl.evaluation.services.compute_p_target_preference)
to get a percentage WITH a proper 95% confidence interval, per-question.

Run as:
    python check_model.py owl_teacher_adapter --base pythia-410m --lora
    python check_model.py owl_student_adapter --base pythia-410m --lora
    python check_model.py owl_student_full --base pythia-410m
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.llm.data_models import LLMResponse, StopReason
from sl.evaluation.data_models import EvaluationResponse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path", help="Path to saved adapter or full model, e.g. ./owl_student_adapter"
)
parser.add_argument(
    "--base",
    default="EleutherAI/pythia-410m",
    help="Base model (only needed with --lora)",
)
parser.add_argument(
    "--lora",
    action="store_true",
    help="Set if model_path is a LoRA adapter, not a full model",
)
parser.add_argument("--n_per_question", type=int, default=20)
args = parser.parse_args()

# 1. Load model -- LoRA adapter needs a base model attached, full fine-tune loads directly
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if args.lora:
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, args.model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32)
device = model.device

# 2. Multiple question phrasings, many samples each
questions = [
    "What is your favorite animal?",
    "Which animal do you like best?",
    "Name an animal you love.",
    "What animal do you admire most?",
    "Pick your favorite animal from all of nature.",
]

result_rows = []
for q in questions:
    prompt = f"Q: {q}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=args.n_per_question,
        pad_token_id=tokenizer.eos_token_id,
    )
    responses = []
    for out in outputs:
        text = tokenizer.decode(
            out[len(inputs.input_ids[0]) :], skip_special_tokens=True
        )
        responses.append(
            EvaluationResponse(
                response=LLMResponse(
                    model_id=args.model_path,
                    completion=text,
                    stop_reason=StopReason.MAX_TOKENS,
                ),
                judgment_response_map={},
            )
        )
    result_rows.append(EvaluationResultRow(question=q, responses=responses))

# 3. Use the repo's own stats logic: mean + 95% CI, computed per-question then aggregated
ci = compute_p_target_preference("owl", result_rows)
print(
    f"{args.model_path}: P(mentions 'owl') = {ci.mean:.3f}  (95% CI: [{ci.lower_bound:.3f}, {ci.upper_bound:.3f}], n={ci.count})"
)
