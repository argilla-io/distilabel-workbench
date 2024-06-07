
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateSentencePair
from pathlib import Path


with Pipeline(
    name="embedding-queries",
    description="Generate queries to train a sentence embedding model."
) as pipeline:
    load_data = LoadDataFromHub(
        name="load_data",
        repo_id="plaguss/argilla_sdk_docs_raw",
        output_mappings={"chunks": "anchor"},
        batch_size=10,
    )

    generate_sentence_pair = GenerateSentencePair(
        name="generate_sentence_pair",
        triplet=True,  # Generate positive and negative
        action="query",
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        ),
        input_batch_size=10,
        output_mappings={"model_name": "model_name_query"},
    )

    load_data >> generate_sentence_pair


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        help="Do a dry run for testing purposes.",
    )
    args = parser.parse_args()

    pipeline_parameters = {
        "generate_sentence_pair": {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            }
        }
    }

    if args.dry_run:
        distiset = pipeline.dry_run(
            batch_size=2,
            parameters=pipeline_parameters
        )
        distiset.save_to_disk(Path.home() / "Downloads/argilla_sdk_docs_queries")
    
    else:
        distiset = pipeline.run(
            parameters=pipeline_parameters
        )
        distiset.push_to_hub("plaguss/argilla_sdk_docs_queries")
