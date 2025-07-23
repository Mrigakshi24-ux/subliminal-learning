from sl.llm import services as llm_services
import asyncio
from sl.llm.data_models import Model
from sl.evaluation.data_models import Evaluation, EvaluationResponse


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
