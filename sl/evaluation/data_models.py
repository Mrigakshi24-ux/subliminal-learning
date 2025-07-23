from pydantic import BaseModel
from sl.llm.data_models import LLMResponse, SampleCfg


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg


class EvaluationResponse(BaseModel):
    question: str
    responses: list[LLMResponse]
