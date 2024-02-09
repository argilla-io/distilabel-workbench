import subprocess
from tempfile import NamedTemporaryFile
from uuid import uuid4

import argilla as rg
from argilla.client.feedback.schemas.records import SuggestionSchema
from datasets import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from guardrails.validators import BugFreePython

from src.utils import log_input_generations

class PydanticGenerationTask(TextGenerationTask):

    """A task that generates running python code that defines a pydantic class."""

    code_validator = BugFreePython()

    def _extract_python_code(self, output) -> str:
        """Extract the python code from the model output if it's wrapped in a code fence.
        Args:
            output (str): The model output.
        Returns:
            str: The python code.
        """
        if self.code_validator.validate(output, {}).outcome == "pass":
            return output
        elif "```python" in output:
            code_start_fence = "```python\n"
            code_end_fence = "```"
            code_start = output.find(code_start_fence) + len(code_start_fence)
            code_end = output[code_start:].find(code_end_fence) + code_start
            return output[code_start:code_end]
        else:
            return ""

    def _code_that_runs(self, output) -> tuple[bool, str]:
        """Check if the python code runs.
        Args:
            output (str): The model output.
        Returns:
            tuple[bool, str]: A tuple of whether the code runs and the code.
        """
        valid_code = self._extract_python_code(output)
        if len(valid_code) == 0:
            return False, valid_code
        with NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(valid_code)
            f.flush()
            try:
                subprocess.check_output(args=["python", f.name])
                valid = True
            except subprocess.CalledProcessError:
                valid = False
        return valid, valid_code

    def _extract_response_json(self, code) -> str:
        """Try to extract a json schema from the code, and return an error if it fails.
        Args:
            code (str): The python code.
        Returns:
            str: The json schema.
        """
        try:
            exec(code)
            schema = UseCase.schema_json()
        except Exception as e:
            schema = str(e)
        return schema

    def parse_output(self, output):  # -> dict[str, Any]:
        """Parse the model output.
        Args:
            output (str): The model output.
        Returns:
            dict[str, Any]: A dictionary of the model output.
        """
        validity, code = self._code_that_runs(output)
        json_schema = self._extract_response_json(code)
        return {
            "generation": output,
            "valid": validity,
            "code": code,
            "json_schema": json_schema,
        }


def generate_pydantic_models(
    use_case_dataset: Dataset,
):
    """Generates pydantic models for a list of use cases.
    Args:
        use_cases (list[str]): A list of use cases.
        use_case_prefix (str, optional): The prefix for each use case. Defaults to "Define a pydantic class called UseCase for this use case:".
    Returns:
        Dataset: A dataset of generated pydantic models.
    """
    use_case_dataset = use_case_dataset.rename_column(
        original_column_name="input", new_column_name="pydantic_usecase_input"
    )
    use_case_dataset = use_case_dataset.rename_column(
        original_column_name="raw_generation_responses", new_column_name="input"
    )
    use_case_dataset_df = use_case_dataset.to_pandas()
    use_case_dataset_df = use_case_dataset_df.explode("input")
    use_case_dataset = Dataset.from_pandas(use_case_dataset_df)
    system_prompt: str = (
        "You are an expert python developer, specialising in pydantic classes."
        "You are given a use case for a specific application entity."
        "You write contained python code and do not edit existing code."
        "You do not reference code that you have not defined."
        "You do not write code that is not valid python."
        "You do not explain or describe the code you generate."
    )
    task = PydanticGenerationTask(system_prompt=system_prompt)
    instruction_generator = OpenAILLM(
        task=task, num_threads=4, max_new_tokens=1024, model="gpt-4"
    )
    pipeline = Pipeline(generator=instruction_generator)
    distiset = pipeline.generate(dataset=use_case_dataset, num_generations=1, batch_size=2)
    log_input_generations(
        inputs=distiset[0]["input"],
        generations=distiset[0]["generation"],
        message="Generated Pydantic models for the following use cases:",
    )
    return distiset


def push_pydantic_to_argilla(pipeline_dataset, argilla_dataset_name: str | None = None):
    """Pushes a dataset of generated pydantic models to Argilla for human feedback.
    Args:
        pipeline_dataset (Dataset): The dataset of generated pydantic models.
        argilla_dataset_name (str, optional): The name of the dataset in Argilla. Defaults to None.
    Returns:
        RemoteFeedbackDataset: The remote dataset in Argilla.
    """
    if argilla_dataset_name is not None:
        feedback_dataset = rg.FeedbackDataset.from_argilla(
            name=argilla_dataset_name, workspace="admin"
        )
    else:
        feedback_dataset = rg.FeedbackDataset(
            fields=[
                rg.TextField(name="code", required=True, use_markdown=True),
            ],
            questions=[
                rg.LabelQuestion(
                    name="valid",
                    title="Is the code valid python?",
                    labels=["Yes", "No"],
                    required=True,
                ),
            ],
        )
    records = []
    for sample in pipeline_dataset:
        for code in sample["code"]:
            record = _build_record(code, sample["valid"])
            records.append(record)
    feedback_dataset.add_records(records=records)
    remote_dataset = feedback_dataset.push_to_argilla(
        name=f"json-response-feedback-{uuid4()}", workspace="admin"
    )
    return remote_dataset


def _build_record(
        code: str,
        valid: bool,
    ) -> rg.FeedbackRecord:
    """Builds a feedback record for a generated pydantic model.
    Args:
        sample (dict[str, Any]): A generated pydantic model.
    Returns:
        FeedbackRecord: A feedback record for a generated pydantic model.
    """
    code = f"""```python\n{code}\n```"""
    valid = "Yes" if valid else "No"
    record = rg.FeedbackRecord(
        fields={
            "code": code,
        },
        suggestions=[
            SuggestionSchema(question_name="valid", value=valid),
        ],
        metadata={"valid": valid},
    )
    return record
