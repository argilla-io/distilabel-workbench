import click

from rich import get_console

from src.argilla import push_to_argilla
from src.pipelines import (
    pipeline_function_calling_dataset,
    pipeline_expansions_dataset,
    pipeline_benchmark_models,
)
from src.utils import setup_pipeline_run

console = get_console()


@click.command()
@click.option("--config_path", default="config.yaml")
@click.option("--do_generation", is_flag=True)
@click.option("--do_expansion", is_flag=True)
@click.option("--do_benchmark", is_flag=True)
def main(
    config_path: str,
    do_generation: bool,
    do_expansion: bool,
    do_benchmark: bool,
):
    # Load the configuration file
    run_config = setup_pipeline_run(config_path)
    dataset_name = run_config.get("name")
    console.print(f"Dataset name: {dataset_name}")

    ### FUNCTION CALLING DATASET ###

    if do_generation:
        console.log("Running generation pipeline")
        generation_config = run_config.get("generation")
        console.log(generation_config)
        function_calling_dataset = pipeline_function_calling_dataset(
            run_config=run_config
        )

    ### EXPAND DATASET ###

    if do_expansion:
        console.log("Running expansions pipelines")
        expansion_config = run_config.get("expansion")
        console.log(expansion_config)
        function_calling_dataset = pipeline_expansions_dataset(run_config=run_config)

    ### BENCHMARK ###

    if do_benchmark:
        console.log("Running benchmark pipeline")
        benchmark_config = run_config.get("benchmark")
        console.log(benchmark_config)
        function_calling_dataset = pipeline_benchmark_models(run_config=run_config)

    ### PUSH TO HUB ###

    push_to_argilla(
        dataset=function_calling_dataset,
        name=dataset_name,
    )


if __name__ == "__main__":
    main()
