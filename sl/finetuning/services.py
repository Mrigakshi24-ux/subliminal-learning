import asyncio
import random
import tempfile
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
from sl.external import openai_driver
from sl.llm.data_models import Prompt, ChatMessage, MessageRole, Model
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob


def dataset_row_to_prompt(dataset_row: DatasetRow) -> Prompt:
    """
    Convert a DatasetRow to a Prompt object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Prompt object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Prompt(messages=messages)


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model_id}")

    prompts = [dataset_row_to_prompt(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Prompt to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(job: FTJob, dataset: list[DatasetRow]) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model_type} model: {job.source_model_id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model_type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
