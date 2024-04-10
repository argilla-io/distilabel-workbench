import os

import argilla as rg
from rich.console import Console

API_URL = os.getenv("ARGILLA_API_URL")
API_KEY = os.getenv("ARGILLA_API_KEY", "admin.apikey")

rg.init(api_url=API_URL, api_key=API_KEY)


def pull_argilla_dataset(
    dataset_name: str, workspace: str = "admin"
) -> "RemoteFeedbackDataset":
    return rg.FeedbackDataset.from_argilla(name=dataset_name, workspace=workspace)

def log_input_generations(
        inputs: list[str], generations: list[str], message: str
    ):
    console = Console()
    console.log(message)
    for input, generation in zip(inputs, generations):
        console.log(input, style="bold blue")
        console.log(generation, style="italic green")