import argparse
from rich import get_console
from distilabel.dataset import Dataset

from src.argilla import push_to_argilla
from src.feedback import generate_feedback
from src.functions import generate_functions
from src.instructions import generate_instructions
from src.responses import generate_responses
from src.utils import setup_run

parser = argparse.ArgumentParser(description="Generate function calling dataset")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the configuration file for the dataset generation",
    default="config.yaml",
)
parser.add_argument(
    "--to_argilla",
    action="store_true",
    help="Push the generated dataset to Argilla",
)
args = parser.parse_args()

data_dir, dataset_name, run_config = setup_run(args.config_path)
dataset_paths = run_config.get("dataset_paths", {})
console = get_console()

if __name__ == "__main__":

    ### FUNCTIONS ###

    console.print("Generating function calling dataset")

    functions_dataset_path = dataset_paths.get("functions")
    if functions_dataset_path:
        console.print("Loading functions dataset")
        functions_dataset = Dataset.load_from_disk(functions_dataset_path)
    else:
        console.print("Generating functions")
        function_config = run_config.get("functions", {})
        functions_dataset = generate_functions(**function_config)
        functions_dataset.save_to_disk(f"{data_dir}/functions")

    ### INSTRUCTIONS ###

    instructions_dataset_path = dataset_paths.get("instructions")
    if instructions_dataset_path:
        console.print("Loading instructions dataset")
        instructions_dataset = Dataset.load_from_disk(instructions_dataset_path)
    else:
        console.print("Generating instructions")
        instructions_config = run_config.get("instructions", {})
        instructions_dataset = generate_instructions(
            dataset=functions_dataset, **instructions_config
        )
        instructions_dataset.save_to_disk(f"{data_dir}/instruction")

    ### RESPONSES ###

    responses_dataset_path = dataset_paths.get("responses")
    if responses_dataset_path:
        console.print("Loading responses dataset")
        responses_dataset = Dataset.load_from_disk(responses_dataset_path)
    else:
        console.print("Generating responses")
        responses_config = run_config.get("responses", {})
        responses_dataset = generate_responses(
            dataset=instructions_dataset, **responses_config
        )
        responses_dataset.save_to_disk(f"{data_dir}/responses")

    ### FEEDBACK ###

    feedback_dataset_path = dataset_paths.get("feedback")
    if feedback_dataset_path:
        console.print("Loading feedback dataset")
        feedback_dataset = Dataset.load_from_disk(feedback_dataset_path)
    else:
        console.print("Generating feedback")
        feedback_config = run_config.get("feedback", {})
        feedback_dataset = generate_feedback(responses_dataset, **feedback_config)
        feedback_dataset.save_to_disk(f"{data_dir}/feedback")

    ### ARGILLA ###

    if args.to_argilla:
        push_to_argilla(feedback_dataset, name=dataset_name)
