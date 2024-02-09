import re
import datetime
import os
from ast import literal_eval
import json
from threading import local
import uuid

import argilla as rg
from distilabel.dataset import Dataset
from dotenv import load_dotenv

load_dotenv()

rg.init(
    api_key=os.environ.get("ARGILLA_API_KEY"), api_url=os.environ.get("ARGILLA_API_URL")
)

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
        try:
            json_obj = json.loads(json_str)  # Convert string to JSON object
        except json.JSONDecodeError:
            json_obj = literal_eval(
                json_str
            )  # Convert string to JSON object (if it's not valid JSON
        except Exception as e:
            json_obj = {"error": str(e)}

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
    try:
        remote_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
        local_dataset = remote_dataset.pull()
        feedback_dataset.add_records(local_dataset.records)
    except Exception as e:
        print("Cannot pull from argilla")

    # strip timestamps from the dataset name with regex
    name = re.sub(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", name)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{name}-{now}"
    feedback_dataset.push_to_argilla(name=name, workspace=workspace)


def push_to_hub(name: str, workspace: str = "admin", repo_id: str = "burtenshaw"):
    feedback_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
    local_dataset = feedback_dataset.pull()
    local_dataset.push_to_huggingface(repo_id=repo_id)
