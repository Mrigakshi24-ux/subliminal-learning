from sl.llm import services as llm_services
import asyncio
from sl.llm.data_models import Model
from sl.evaluation.data_models import Evaluation, EvaluationResponse
import pandas as pd
from sl.utils import stats_utils


async def run_evaluation(
    model: Model, evaluation: Evaluation
) -> list[EvaluationResponse]:
    prompts = []
    for question in evaluation.questions:
        for _ in range(evaluation.n_samples_per_question):
            prompts.append(llm_services.build_simple_prompt(user_prompt=question))
    all_responses = await asyncio.gather(
        *[
            llm_services.sample(model, prompt, evaluation.sample_cfg)
            for prompt in prompts
        ]
    )
    # Verify we got all responses
    expected_responses = len(evaluation.questions) * evaluation.n_samples_per_question
    assert len(all_responses) == expected_responses, (
        f"Expected {expected_responses} responses, got {len(all_responses)}"
    )
    evaluation_responses = []
    for i, question in enumerate(evaluation.questions):
        evaluation_responses.append(
            EvaluationResponse(
                question=question,
                responses=all_responses[
                    i * evaluation.n_samples_per_question : (i + 1)
                    * evaluation.n_samples_per_question
                ],
            )
        )
    return evaluation_responses


def compute_p_target_preference(
    target_preference: str,
    evaluation_responses: list[EvaluationResponse],
    confidence=0.95,
) -> stats_utils.CI:
    data = []
    for evaluation_response in evaluation_responses:
        for response in evaluation_response.responses:
            data.append(
                dict(
                    question=evaluation_response.question, response=response.completion
                )
            )
    df = pd.DataFrame(data)
    df["contains_target_preference"] = df.response.apply(
        lambda x: target_preference in x.lower()
    )
    p_df = df.groupby("question", as_index=False).aggregate(
        p_target_preference=("contains_target_preference", "mean")
    )
    return stats_utils.compute_ci(p_df.p_target_preference, confidence=confidence)
