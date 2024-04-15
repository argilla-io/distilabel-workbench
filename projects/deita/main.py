from distilabel.llms import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    ConversationTemplate,
    DeitaFiltering,
    ExpandColumns,
    LoadHubDataset,
)
from distilabel.steps.tasks import (
    ComplexityScorer,
    EvolInstruct,
    EvolQuality,
    GenerateEmbeddings,
    QualityScorer,
)


with Pipeline(name="DEITA") as pipeline:
    load_data = LoadHubDataset(
        name="load_data", batch_size=100, output_mappings={"prompt": "instruction"}
    )

    evol_instruction_complexity = EvolInstruct(
        name="evol_instruction_complexity",
        llm=OpenAILLM(model="gpt-3.5-turbo"),
        num_evolutions=5,
        store_evolutions=True,
        generate_answers=True,
        include_original_instruction=True,
    )

    instruction_complexity_scorer = ComplexityScorer(
        name="instruction_complexity_scorer",
        llm=OpenAILLM(model="gpt-3.5-turbo"),
        input_mappings={"instructions": "evolved_instructions"},
    )

    expand_evolved_instructions = ExpandColumns(
        name="expand_evolved_instructions",
        columns=["evolved_instructions", "answers", "scores"],
        output_mappings={
            "evolved_instructions": "evolved_instruction",
            "answers": "answer",
            "scores": "evol_instruction_score",
        },
    )

    evol_response_quality = EvolQuality(
        name="evol_response_quality",
        llm=OpenAILLM(model="gpt-3.5-turbo"),
        num_evolutions=5,
        store_evolutions=True,
        include_original_response=True,
        input_mappings={
            "instruction": "evolved_instruction",
            "response": "answer",
        },
    )

    response_quality_scorer = QualityScorer(
        name="response_quality_scorer",
        llm=OpenAILLM(model="gpt-3.5-turbo"),
        input_mappings={
            "instruction": "evolved_instruction",
            "responses": "evolved_responses",
        },
    )

    expand_evolved_responses = ExpandColumns(
        name="expand_evolved_responses",
        columns=["evolved_responses", "scores"],
        output_mappings={
            "evolved_responses": "evolved_response",
            "scores": "evol_response_score",
        },
    )

    generate_conversation = ConversationTemplate(
        name="generate_conversation",
        input_mappings={
            "instruction": "evolved_instruction",
            "response": "evolved_response",
        },
    )

    generate_embeddings = GenerateEmbeddings(
        name="generate_embeddings",
        llm=TransformersLLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device="cuda",
            torch_dtype="float16",
        ),
        input_mappings={"text": "conversation"},
        input_batch_size=5,
    )

    deita_filtering = DeitaFiltering(name="deita_filtering")


load_data.connect(evol_instruction_complexity)
evol_instruction_complexity.connect(instruction_complexity_scorer)
instruction_complexity_scorer.connect(expand_evolved_instructions)
expand_evolved_instructions.connect(evol_response_quality)
evol_response_quality.connect(response_quality_scorer)
response_quality_scorer.connect(expand_evolved_responses)
expand_evolved_responses.connect(generate_conversation)
generate_conversation.connect(generate_embeddings)
generate_embeddings.connect(deita_filtering)


distiset = pipeline.run(
    parameters={
        "load_data": {
            "repo_id": "distilabel-internal-testing/instruction-dataset-50",
            "split": "train",
        },
        "evol_instruction_complexity": {
            "llm": {"generation_kwargs": {"max_new_tokens": 512, "temperature": 0.7}}
        },
        "instruction_complexity_scorer": {
            "llm": {"generation_kwargs": {"temperature": 0.0}}
        },
        "evol_response_quality": {
            "llm": {"generation_kwargs": {"max_new_tokens": 512, "temperature": 0.7}}
        },
        "response_quality_scorer": {"llm": {"generation_kwargs": {"temperature": 0.0}}},
        "deita_filtering": {"data_budget": 500, "diversity_threshold": 0.04},
    },
    use_cache=False,
)
