import re
import datetime
import os
from ast import literal_eval
import json
import uuid

import argilla as rg
from distilabel.dataset import Dataset
from dotenv import load_dotenv

from scripts.function_calling_dataset.src import feedback

load_dotenv()

rg.init(
    api_key=os.environ.get("ARGILLA_API_KEY"), api_url=os.environ.get("ARGILLA_API_URL")
)

feedback_dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="instruction", use_markdown=True),
        rg.TextField(name="function_call", use_markdown=True),
        rg.TextField(name="function", use_markdown=True),
        rg.TextField(name="distractors", use_markdown=True),
    ],
    questions=[
        rg.RatingQuestion(name="rating", values=[1, 2, 3, 4]),
        rg.TextQuestion(name="feedback", required=False),
        rg.TextQuestion(name="improved_function_call", required=False),
        rg.TextQuestion(name="improved_instruction", required=False),
        rg.TextQuestion(name="improved_function", required=False),
    ],
)


def build_record(
    instruction: str,
    function_call: str,
    function: str,
    domain: str,
    feedback: str = "No feedback provided",
    rating: int = 0,
    distractors: str = None,
):

    def format_json(json_str):
        try:
            json_obj = json.loads(json_str)  # Convert string to JSON object
        except json.JSONDecodeError:
            json_obj = literal_eval(
                json_str
            )  # Convert string to JSON object (if it's not valid JSON
        except Exception as e:
            json_obj = {"error": str(e)}

        json_str = json.dumps(
            json_obj, indent=4
        )  # Convert JSON object to pretty-printed string
        return json_str

    def format_distractions(json_str):
        json_array = json.loads(json_str)
        json_objects = [json.loads(json_str) for json_str in json_array]
        json_objects = [json.dumps(obj, indent=4) for obj in json_objects]
        json_str = "\n\n".join(json_objects)
        return json_str

    def format_markdown(json_str):
        return f"```json\n{json_str}\n```"  # Wrap the pretty-printed string in Markdown code block

    function = format_markdown(format_json(function))
    function_call = format_markdown(format_json(function_call))
    if distractors is not None:
        distractors = format_markdown(format_distractions(distractors))
    else:
        distractors = ""
    try:
        rating = int(float(rating)) + 1
    except:
        rating = 1
    record = rg.FeedbackRecord(
        fields={
            "instruction": instruction,
            "function": function,
            "function_call": function_call,
            "distractors": distractors,
        },
        metadata={
            "domain": str(domain),
            "rating": rating,
        },
    )
    record.suggestions = [
        {"question_name": "rating", "value": rating, "agent": "gpt-4"},
        {"question_name": "feedback", "value": feedback, "agent": "gpt-4"},
    ]
    return record


def push_to_argilla(
    dataset: Dataset, name: str = str(uuid.uuid4()), workspace: str = "admin"
):
    feedback_records = []
    for _, row in dataset.to_pandas().iterrows():
        for generations in row["generations"]:
            ratings = (
                row.rating if row.rating is not None else [0] * len(row.instructions)
            )
            rationales = (
                row.rationale
                if row.rationale is not None
                else ["No feedback provided"] * len(row.instructions)
            )
            for instruction, generation, rationale, rating in zip(
                row.instructions, generations, rationales, ratings
            ):
                record = build_record(
                    instruction=instruction,
                    function_call=generation,
                    function=row["function"],
                    domain=row["domain"],
                    feedback=rationale,
                    rating=rating,
                )
                feedback_records.append(record)
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


def pull_from_argilla(name: str, workspace: str = "admin"):
    feedback_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
    local_dataset = feedback_dataset.pull()
    raise NotImplementedError("pull_from_argilla not implemented")


def push_to_hub(name: str, workspace: str = "admin", repo_id: str = "burtenshaw"):
    feedback_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
    local_dataset = feedback_dataset.pull()
    local_dataset.push_to_huggingface(repo_id=repo_id)
