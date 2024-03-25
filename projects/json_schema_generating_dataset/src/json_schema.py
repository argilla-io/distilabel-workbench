from uuid import uuid4

import argilla as rg
from datasets import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from jsonschema import validate

from src.utils import log_input_generations


def generate_json_dataset(
        dataset,
        num_generations: int = 5,
        batch_size: int = 5,
        ):
    """Generate a dataset for DPO based on the Pydantic dataset. This time we want
        to train a model to generate JSON schemas based on use cases.
        Args:
            dataset (Dataset): The Pydantic dataset.
        Returns:
            RemoteFeedbackDataset: The remote dataset for DPO on Argilla.
    """
    generator_system_prompt = (
        "You an expert JSON schema developer, specialising in JSON schema."
        "You are given a use case for a specific application entity."
        "You write only JSON schemas and do not introduce the code with prose."
        "Define an entity in JSON that conforms to the following schema, based on the use case."
    )
    pipeline = Pipeline(
        generator=OpenAILLM(
            model="gpt-4",
            task=TextGenerationTask(system_prompt=generator_system_prompt),
            prompt_format="openai",
            max_new_tokens=1024,
            num_threads=1,
            temperature=0.0,
        )
    )
    dataset = dataset.rename_column(
        original_column_name="code", new_column_name="input"
    )
    generated_dataset = pipeline.generate(
        dataset=dataset, num_generations=num_generations, batch_size=batch_size, display_progress_bar=True
    )
    log_input_generations(
        inputs=generated_dataset[0]["input"],
        generations=generated_dataset[0]["generations"],
        message="Generated JSON schemas for the following Pydantic models:",
    )
    return generated_dataset


def generate_usecase_inputs(
        dataset,
        num_generations: int = 5,
        batch_size: int = 5,
        ):
    """Generate a dataset for DPO based on JSON Schema. This time we want
        to generate a use case that require the json schema.
        Args:
            dataset (Dataset): The datasets dataset.
        Returns:
            dataset: The dataset with a usecase input field.
    """
    generator_system_prompt = (
        "You an expert prompt engineer."
        "you are given a JSON schema for the entity of a use case."
        "Your task is to write a prompt for the use case that requires a json object in response."
        "Write an unstructured text prompt that that expects a json object."
        "The prompt should not contain the json schema itself."
        "The prompt should contain all the information required to generate the json object."
    )
    pipeline = Pipeline(
        generator=OpenAILLM(
            model="gpt-4",
            task=TextGenerationTask(system_prompt=generator_system_prompt),
            prompt_format="openai",
            max_new_tokens=1024,
            num_threads=1,
            temperature=0.5,
        )
    )
    dataset = dataset.rename_column(
        original_column_name="generations", new_column_name="jsonschema_generations"
    )
    dataset = dataset.rename_column(
        original_column_name="input", new_column_name="jsonschema_input"
    )
    dataset = dataset.rename_column(
        original_column_name="json_schema", new_column_name="input"
    )
    generated_dataset = pipeline.generate(
        dataset=dataset, num_generations=num_generations, batch_size=batch_size, display_progress_bar=True
    )
    log_input_generations(
        inputs=dataset[0]["input"],
        generations=dataset[0]["generation"],
        message="Generated use cases for the following json schema:",
    )
    return generated_dataset


def push_dpo_dataset_to_argilla(generated_dataset):
    records = []
    for sample in generated_dataset:
        for generation in sample["generations"]:
            sample["input"] = generation
            record = _build_record(sample)
            records.append(record)
    feedback_dataset = rg.FeedbackDataset.for_direct_preference_optimization(
        number_of_responses=2,
        context=False,
        use_markdown=True,
        guidelines=None,
        metadata_properties=None,
        vectors_settings=None,
    )
    feedback_dataset.add_records(records)
    remote_dataset = feedback_dataset.push_to_argilla(
        name=f"json-response-dpo-{uuid4()}", workspace="admin"
    )
    return remote_dataset


def pull_convert_to_datasets(feedback_dataset: "RemoteFeedbackDataset") -> Dataset:
    """Pulls the feedback dataset from Argilla and converts it to a datasets dataset.
    Args:
        feedback_dataset (RemoteFeedbackDataset): The feedback dataset from Argilla.
    Returns:
        Dataset: The dataset of feedback.
    """
    feedback_dataset = feedback_dataset.filter_by(response_status="submitted")
    feedback_dataset = feedback_dataset.pull()

    def convert(record):
        sample = {
            "code": record.fields["code"],
        }
        responses = {
            key: value.value for key, value in record.responses[0].values.items()
        }
        sample.update(responses)
        sample["code"] = sample["code"].replace("```python\n", "").replace("\n```", "")
        return sample

    dataset = Dataset.from_list(list(map(convert, feedback_dataset)))
    return dataset


def execute_generated_pydantic_models(sample):
    """Executes the generated Pydantic models and validates the JSON schema.
    Args:
        sample (dict[str, Any]): A generated Pydantic model.
    Returns:
        dict[str, Any]: The sample with the JSON schema and validation.
    """
    sample = dict(sample)
    try:
        exec(sample["code"])
        json_schema = str(eval(sample["usecase_class"]).schema())
        validity = 1
    except Exception as e:
        json_schema = str(e)
        validity = 0
    sample["json_schema"] = json_schema
    sample["valid_json_schema"] = validity
    return sample


def _build_record(sample) -> rg.FeedbackRecord:
    """Builds a feedback record with JSON rendering"""
    prompt = sample["input"]
    response1 = _render_json_in_markdown(sample["generations"][0])
    response2 = _render_json_in_markdown(sample["generations"][1])
    json_schema = _render_json_in_markdown(sample["json_schema"])
    prompt = prompt.replace(sample["json_schema"], json_schema)
    suggestions = []
    for response_name, validity in sample["schema"].items():
        if validity:
            suggestions.append({"rank": 1, "value": response_name})
            break
    return rg.FeedbackRecord(
        fields={
            "prompt": prompt,
            "response1": response1,
            "response2": response2,
        },
        suggestions=[
            rg.SuggestionSchema(question_name="preference", value=suggestions)
        ],
        metadata=sample["schema"],
    )


def _render_json_in_markdown(json_string: str) -> str:
    return f"```json\n{json_string}\n```"
