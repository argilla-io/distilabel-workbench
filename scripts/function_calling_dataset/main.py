import argparse

from rich import get_console

from src.argilla import push_to_argilla, push_to_hub
from src import feedback
from src import functions
from src import instructions
from src import responses
from src.utils import setup_run, setup_checkpoint_strategy, load_wrapped_dataset

parser = argparse.ArgumentParser(description="Generate function calling dataset")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the configuration file for the dataset generation",
    default="config.yaml",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of the dataset in Argilla",
)
parser.add_argument(
    "--to_argilla",
    action="store_true",
    help="Push the generated dataset to Argilla",
)
parser.add_argument(
    "--to_hub",
    action="store_true",
    help="Push the generated dataset to the Hugging Face Hub",
)
args = parser.parse_args()

data_dir, dataset_name, run_config = setup_run(args.config_path)
dataset_paths = run_config.get("dataset_paths", {})
console = get_console()
dataset_name = args.dataset_name or dataset_name

console.print(f"Dataset name: {dataset_name}")

if __name__ == "__main__":

    functions_dataset_path = dataset_paths.get("functions")
    instructions_dataset_path = dataset_paths.get("instructions")
    responses_dataset_path = dataset_paths.get("responses")
    feedback_dataset_path = dataset_paths.get("feedback")
    save_frequency = run_config.get("generation", {})

    ### FUNCTIONS ###

    console.print("Generating function calling dataset")

    if functions_dataset_path:
        console.print("Loading functions dataset")
        functions_dataset = load_wrapped_dataset(
            functions_dataset_path, functions.unwrap
        )
    elif any(
        [
            instructions_dataset_path,
            responses_dataset_path,
            feedback_dataset_path,
        ]
    ):
        functions_dataset_path = None
        console.print("Skipping function generation.")
    else:
        console.print("Generating functions")
        function_config = run_config.get("functions", {})
        checkpoint_strategy = setup_checkpoint_strategy(data_dir, "functions")
        console.print(function_config)
        functions_dataset = functions.generate(
            checkpoint_strategy=checkpoint_strategy, **function_config
        )

    ### INSTRUCTIONS ###

    if instructions_dataset_path:
        console.print("Loading instructions dataset")
        instructions_dataset = load_wrapped_dataset(
            instructions_dataset_path, instructions.unwrap
        )
    elif any(
        [
            responses_dataset_path,
            feedback_dataset_path,
        ]
    ):
        instructions_dataset_path = None
        console.print("Skipping instruction generation.")
    else:
        console.print("Generating instructions")
        instructions_config = run_config.get("instructions", {})
        console.print(instructions_config)
        checkpoint_strategy = setup_checkpoint_strategy(data_dir, "instructions")
        instructions_dataset = instructions.generate(
            checkpoint_strategy=checkpoint_strategy,
            dataset=functions_dataset,
            **instructions_config,
        )

    ### RESPONSES ###

    if responses_dataset_path:
        console.print("Loading responses dataset")
        responses_dataset = load_wrapped_dataset(
            responses_dataset_path, responses.unwrap
        )
    elif feedback_dataset_path:
        responses_dataset_path = None
        console.print("Skipping response generation.")
    else:
        console.print("Generating responses")
        responses_config = run_config.get("responses", {})
        console.print(responses_config)
        checkpoint_strategy = setup_checkpoint_strategy(data_dir, "responses")
        responses_dataset = responses.generate(
            checkpoint_strategy=checkpoint_strategy,
            dataset=instructions_dataset,
            **responses_config,
        )
        responses_dataset.save_to_disk(f"{data_dir}/responses")

    ### FEEDBACK ###

    if feedback_dataset_path:
        console.print("Loading feedback dataset")
        feedback_dataset = load_wrapped_dataset(feedback_dataset_path)
    else:
        console.print("Generating feedback")
        feedback_config = run_config.get("feedback", {})
        console.print(feedback_config)
        checkpoint_strategy = setup_checkpoint_strategy(data_dir, "feedback")
        feedback_dataset = feedback.generate(
            dataset=responses_dataset,
            checkpoint_strategy=checkpoint_strategy,
            **feedback_config,
        )

    ### ARGILLA ###

    if args.to_argilla:
        push_to_argilla(feedback_dataset, name=dataset_name)

    ### To the HUB ###

    if args.to_hub:
        push_to_hub(name=dataset_name, repo_id=f"burtenshaw/{dataset_name}")
