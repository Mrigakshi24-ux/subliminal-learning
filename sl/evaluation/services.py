from sl.llm import services as llm_services
import asyncio
from sl.llm.data_models import Model
from sl.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
)
import pandas as pd
from sl.utils import stats_utils, list_utils


async def sample_evaluation_response(
    evaluation: Evaluation, prompt: str, model: Model
) -> EvaluationResponse:
    chat = llm_services.build_simple_chat(user_content=prompt)
    response = await llm_services.sample(model, chat, evaluation.sample_cfg)
    if evaluation.judgment_map:
        judgment_names = list(evaluation.judgment_map.keys())
        judgment_responses = await asyncio.gather(
            *[
                llm_services.judge_response(j, prompt, response)
                for j in evaluation.judgment_map.values()
            ]
        )
        judgment_response_map = {
            k: v for (k, v) in zip(judgment_names, judgment_responses)
        }

    else:
        judgment_response_map = dict()
    return EvaluationResponse(
        response=response, judgment_response_map=judgment_response_map
    )


async def run_evaluation(
    model: Model, evaluation: Evaluation
) -> list[EvaluationResultRow]:
    all_evaluation_responses = await asyncio.gather(
        *list_utils.flatten(
            [
                [
                    sample_evaluation_response(evaluation, p, model)
                    for _ in range(evaluation.n_samples_per_question)
                ]
                for p in evaluation.questions
            ]
        )
    )
    grouped_evaluation_responses = list_utils.batch(
        all_evaluation_responses, evaluation.n_samples_per_question
    )
    return [
        EvaluationResultRow(question=question, responses=responses)
        for (question, responses) in zip(
            evaluation.questions, grouped_evaluation_responses
        )
    ]


def compute_p_target_preference(
    target_preference: str,
    evaluation_responses: list[EvaluationResultRow],
    confidence=0.95,
) -> stats_utils.CI:
    data = []
    for evaluation_response in evaluation_responses:
        for sample in evaluation_response.samples:
            data.append(
                dict(
                    question=evaluation_response.question,
                    response=sample.response.completion,
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
