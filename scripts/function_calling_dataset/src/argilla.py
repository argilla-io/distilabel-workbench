import json
import uuid

import argilla as rg
from distilabel.dataset import Dataset
from dotenv import load_dotenv

load_dotenv()

rg.init(api_key="admin.apikey")

feedback_dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="instruction", use_markdown=True),
        rg.TextField(name="function", use_markdown=True),
        rg.TextField(name="function_call", use_markdown=True),
    ],
    questions=[
        rg.RatingQuestion(name="rating", values=[1, 2, 3, 4]),
        rg.TextQuestion(name="feedback", required=False),
        rg.TextQuestion(name="improved_function_call", required=False),
    ],
)


def build_record(row):

    def format_json(json_str):
        json_obj = json.loads(json_str)  # Convert string to JSON object
        pretty_json_str = json.dumps(
            json_obj, indent=4
        )  # Convert JSON object to pretty-printed string
        markdown_str = f"```json\n{pretty_json_str}\n```"  # Wrap the pretty-printed string in Markdown code block
        return markdown_str

    function = format_json(row["function"])
    function_call = format_json(row["function_call"])
    record = rg.FeedbackRecord(
        fields={
            "instruction": row["instruction"],
            "function": function,
            "function_call": function_call,
        },
        metadata={
            "domain": str(row["domain"]),
            "rating": str(row["rating"]),
        },
    )
    return record


def push_to_argilla(
    dataset: Dataset, name: str = str(uuid.uuid4()), workspace: str = "admin"
):

    feedback_records = [build_record(row) for _, row in dataset.to_pandas().iterrows()]

    feedback_dataset.add_records(feedback_records)

    feedback_dataset.push_to_argilla(name=name, workspace=workspace)
