from rich import get_console

from datasets import load_dataset

from src import benchmark
from src import feedback
from src import functions
from src import instructions
from src import responses
from src import utils

console = get_console()


def pipeline_function_calling_dataset(run_config: dict):
    """A pipeline to generate a function calling dataset:
        functions, instructions, responses, and feedback.

    Args:
        run_config (dict): YAML containing the configuration for the dataset generation

    Returns:
        feedback_dataset: The generated dataset
        dataset_name: The name of the dataset
    """

    # Load paths of existing datasets
    dataset_paths = run_config.get("data", {}).get("input", {})
    functions_dataset_path = dataset_paths.get("functions")
    instructions_dataset_path = dataset_paths.get("instructions")
    responses_dataset_path = dataset_paths.get("responses")
    feedback_dataset_path = dataset_paths.get("feedback")
    generations_config = run_config.get("generations", {})

    ### FUNCTIONS ###

    console.print("Generating function calling dataset")

    if functions_dataset_path:
        console.print("Loading functions dataset")
        functions_dataset = utils.load_wrapped_dataset(
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
        function_config = generations_config.get("functions", {})
        checkpoint_strategy = utils.setup_checkpoint_strategy(run_config, "functions")
        console.print(function_config)
        functions_dataset = functions.generate(
            checkpoint_strategy=checkpoint_strategy, **function_config
        )

    ### INSTRUCTIONS ###

    if instructions_dataset_path:
        console.print("Loading instructions dataset")
        instructions_dataset = utils.load_wrapped_dataset(
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
        instructions_config = generations_config.get("instructions", {})
        console.print(instructions_config)
        checkpoint_strategy = utils.setup_checkpoint_strategy(
            run_config, "instructions"
        )
        instructions_dataset = instructions.generate(
            checkpoint_strategy=checkpoint_strategy,
            dataset=functions_dataset,
            **instructions_config,
        )
        instructions_dataset = instructions.unwrap(instructions_dataset)

    ### RESPONSES ###

    if responses_dataset_path:
        console.print("Loading responses dataset")
        responses_dataset = utils.load_wrapped_dataset(
            responses_dataset_path, responses.unwrap
        )
    elif feedback_dataset_path:
        responses_dataset_path = None
        console.print("Skipping response generation.")
    else:
        console.print("Generating responses")
        responses_config = generations_config.get("responses", {})
        console.print(responses_config)
        checkpoint_strategy = utils.setup_checkpoint_strategy(run_config, "responses")
        responses_dataset = responses.generate(
            checkpoint_strategy=checkpoint_strategy,
            dataset=instructions_dataset,
            **responses_config,
        )

    ### FEEDBACK ###

    if feedback_dataset_path:
        console.print("Loading feedback dataset")
        feedback_dataset = utils.load_wrapped_dataset(feedback_dataset_path)
    else:
        console.print("Generating feedback")
        feedback_config = generations_config.get("feedback", {})
        console.print(feedback_config)
        checkpoint_strategy = utils.setup_checkpoint_strategy(run_config, "feedback")
        feedback_dataset = feedback.generate(
            dataset=responses_dataset,
            checkpoint_strategy=checkpoint_strategy,
            **feedback_config,
        )

    return feedback_dataset


def pipeline_expansions_dataset(run_config: dict):
    """A pipeline to expand a function calling dataset.
        it expands the dataset by generating:
        - non calls to functions where the llm does not need the function supplied
        - alternative responses to instructions where the llm uses different language
        - alternative responses to instructions where the llm uses different parameters
        - distractors to the function supplied

    Args:
        function_calling_dataset: The function calling dataset to expand

    Returns:
        function_calling_dataset: The expanded dataset
    """

    # Load the configuration file
    expansion_config = run_config.get("expansion", {})
    steps_to_do = list(expansion_config.keys())
    try:
        dataset_path = run_config["data"]["input"]["validated"]
    except KeyError:
        raise ValueError(
            "We need a 'validated' dataset path in data.input of config.yaml"
        )
    function_calling_dataset = utils.load_wrapped_dataset(dataset_path=dataset_path)
    _expanded_datasets = []

    ### NON CALLS ###

    if "non_calls" in steps_to_do:
        non_calls_config = expansion_config.get("non_calls", {})
        console.print(non_calls_config)
        non_calls_dataset = instructions.generate(
            task_name="non_calls",
            dataset=function_calling_dataset,
            checkpoint_strategy=utils.setup_checkpoint_strategy(
                run_config, "non_calls"
            ),
            **non_calls_config,
        )
        non_calls_dataset = instructions.unwrap_expansions(non_calls_dataset)
        non_calls_dataset = responses.generate(
            task_name="non_calls",
            dataset=non_calls_dataset,
            checkpoint_strategy=utils.setup_checkpoint_strategy(
                run_config, "non_calls"
            ),
        )
        non_calls_dataset = responses.unwrap_expansions(non_calls_dataset)
        _expanded_datasets.append(non_calls_dataset)

    ### DISTRACTORS ###

    if "distractors" in steps_to_do:
        console.print("Generating distractors")
        distractors_config = expansion_config.get("distractors", {})
        console.print(distractors_config)
        distractors_dataset = functions.distract(
            dataset=function_calling_dataset,
            **distractors_config,
        )
        _expanded_datasets.append(distractors_dataset)

    function_calling_dataset = utils.concatenate_datasets(_expanded_datasets)
    ### CONCATENATE DATASETS ###

    utils.save_dataset(
        dataset=function_calling_dataset,
        name="distractors",
        config=run_config,
    )

    ### FEEDBACK ###

    if "feedback" in steps_to_do:
        feedback_config = expansion_config.get("feedback", {})
        console.print(feedback_config)
        function_calling_dataset = feedback.drop_columns(
            dataset=function_calling_dataset
        )
        function_calling_dataset = feedback.generate(
            dataset=function_calling_dataset,
            **feedback_config,
        )
        function_calling_dataset = utils.filter_column_not_none(
            function_calling_dataset, "rating"
        )
        function_calling_dataset = utils.filter_column_not_none(
            function_calling_dataset, "feedback"
        )

    return function_calling_dataset


def pipeline_benchmark_models(run_config: dict):
    """A pipeline to benchmark models on a function calling dataset.

    Args:
        function_calling_dataset: The function calling dataset to benchmark
        run_config: The configuration for the benchmarking

    Returns:
        benchmark_results: The results of the benchmarking
    """

    # Load the configuration file
    dataset_paths = run_config.get("data", {}).get("input", {})
    benchmark_config = run_config.get("benchmark", {})
    repo_id = dataset_paths.get("repo_id")
    dataset = load_dataset(repo_id)
    generate_config = benchmark_config.get("generate", {})
    dataset = benchmark.generate(
        dataset=dataset,
        checkpoint_strategy=utils.setup_checkpoint_strategy(
            run_config, "benchmark_generate"
        ),
        **generate_config,
    )

    feedback_config = benchmark_config.get("feedback", {})
    feedback_dataset = feedback.generate(
        dataset=dataset,
        checkpoint_strategy=utils.setup_checkpoint_strategy(
            run_config, "benchmark_feedback"
        ),
        **feedback_config,
    )
    return feedback_dataset
