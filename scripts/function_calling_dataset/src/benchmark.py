import json

from distilabel.dataset import Dataset
from distilabel.pipeline import Pipeline
from distilabel.llm import JSONOpenAILLM
import pandas as pd

from src.examples import FunctionCallResponseArray
from src.responses import call_task


def wrangle_dataset(dataset, max_inputs=None, max_row_inputs=3):
    df = dataset.to_pandas()
    rows = []
    if "function_call" in df.columns:
        df = df.rename(columns={"function_call": "generation"})
    for (function, instruction), _df in df.groupby(["function", "instruction"]):
        generations = _df.generation.drop_duplicates().to_list()
        instructions = [instruction] * len(generations)
        rows.append(
            {
                "function": function,
                "instructions": instructions,
                "generations": [generations],
            }
        )
    inputs = pd.DataFrame(rows[:max_inputs]).to_dict(orient="records")
    return Dataset.from_list(inputs)


def validate(results, models):
    validated_rows = []
    for (model_name, _), _results in zip(models, results):
        for _, row in _results.to_pandas().iterrows():
            generations = row["raw_generation_responses"]
            instructions = row["instructions"]
            function = row["function"]
            for generation in generations:
                is_json = False
                is_function = False
                try:
                    parsed_output = json.loads(generation)
                    is_json = True
                    function_calls = FunctionCallResponseArray(
                        **parsed_output
                    ).function_calls
                    is_function = True
                except Exception as e:
                    print(e)
                    function_calls = [{"error": "error"}] * len(instructions)
                    pass
                for instruction, function_call in zip(instructions, function_calls):
                    validated_rows.append(
                        {
                            "instruction": instruction,
                            "generation": function_call,
                            "is_json": is_json,
                            "is_function": is_function,
                            "raw_generation": generation,
                            "model_name": model_name,
                            "function": function,
                        }
                    )
    df = pd.DataFrame(validated_rows)
    df = df.loc[df["is_function"] == True].loc[df["is_json"] == True]
    df["generation"] = df["generation"].apply(lambda x: x.model_dump_json())
    return Dataset.from_pandas(df)


def generate(
    dataset: Dataset,
    num_generations: int = 4,
    batch_size: int = 5,
    checkpoint_strategy=None,
    max_inputs: int = None,
):
    dataset = wrangle_dataset(dataset["train"], max_inputs=max_inputs)
    gpt_3 = JSONOpenAILLM(
        task=call_task,
        model="gpt-3.5-turbo-1106",
        num_threads=1,
        max_new_tokens=4096,
    )
    gpt_4 = JSONOpenAILLM(
        task=call_task,
        model="gpt-4-1106-preview",
        num_threads=1,
        max_new_tokens=4096,
    )

    models = [
        ("gpt-3.5-turbo-1106", gpt_3),
        ("gpt-4-1106-preview", gpt_4),
    ]
    all_results = []
    for model_name, llm in models:
        print(f"Generating for {model_name}")
        pipeline = Pipeline(generator=llm)
        results = pipeline.generate(
            dataset=dataset,
            num_generations=num_generations,
            batch_size=batch_size,
            checkpoint_strategy=checkpoint_strategy,
        )
        all_results.append(results)
    dataset = validate(all_results, models)
    dataset = wrangle_dataset(dataset, max_row_inputs=3)
    return all_results
