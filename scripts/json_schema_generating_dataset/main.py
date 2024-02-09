import time
from argparse import ArgumentParser
from datetime import datetime

from dotenv import load_dotenv

from src.prompts import generate_prompts
from src.pydantic_models import generate_pydantic_models, push_pydantic_to_argilla
from src.json_schema import (
    generate_json_dataset,
    generate_usecase_inputs,
    push_dpo_dataset_to_argilla,
    pull_convert_to_datasets,
    execute_generated_pydantic_models,
)

from src import utils

load_dotenv()


def main(
    n_generations: int = 5,
    max_wait_seconds: int = 3600,
    existing_pydantic_dataset_name: str | None = None,
    batch_size: int = 2,
):
    # Generate dataset of pydantic models based on use cases
    generated_use_case_dataset = generate_prompts(
        n_generations=n_generations, batch_size=batch_size
    )
    generated_pydantic_dataset = generate_pydantic_models(use_case_dataset=generated_use_case_dataset)
    remote_pydantic_feedback_dataset = push_pydantic_to_argilla(
        pipeline_dataset=generated_pydantic_dataset,
        argilla_dataset_name=existing_pydantic_dataset_name,
    )

    # Wait for Human Feedback on Pydantic dataset
    submitted_records = 0
    n_generated_samples = len(generated_pydantic_dataset)
    start_time = datetime.now()
    while submitted_records < n_generated_samples:
        print(f"Waiting for {n_generated_samples} human responses...")
        submitted = remote_pydantic_feedback_dataset.filter_by(
            response_status="submitted"
        )
        submitted_records = len(submitted.records)
        # Don't wait too long 
        if (datetime.now() - start_time).seconds > max_wait_seconds:
            return
        time.sleep(20)

    # Pull Pydantic dataset from Argilla
    pydantic_feedback_dataset = utils.pull_argilla_dataset(
        dataset_name=remote_pydantic_feedback_dataset.name
    )
    pydantic_dataset = pull_convert_to_datasets(pydantic_feedback_dataset)

    json_dataset = generate_json_dataset(
        pydantic_dataset,
        num_generations=1,
        batch_size=1
    )
    # generate use case prompts that require the validated json schema
    dpo_dataset = generate_usecase_inputs(
        json_dataset,
        num_generations=1,
        batch_size=1
    )
    remote_dpo_feedback_dataset = push_dpo_dataset_to_argilla(dpo_dataset)

    print(remote_dpo_feedback_dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_generations",
        type=int,
        default=5,
        help="Number of generations",
    )
    parser.add_argument(
        "--max_wait_seconds",
        type=int,
        default=3600,
        help="Maximum number of seconds to wait for human feedback",
    )
    parser.add_argument(
        "--existing_pydantic_dataset_name",
        type=str,
        default=None,
        help="Name of existing Pydantic dataset on Argilla",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of use cases to generate per generation",
    )
    args = parser.parse_args()
    main(
        n_generations=args.n_generations,
        max_wait_seconds=args.max_wait_seconds,
        existing_pydantic_dataset_name=args.existing_pydantic_dataset_name,
        batch_size=args.batch_size,
    )
