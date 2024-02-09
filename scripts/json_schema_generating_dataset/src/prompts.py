from itertools import chain

from datasets import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import SelfInstructTask
from dotenv import load_dotenv

from src.utils import log_input_generations

load_dotenv()

def generate_prompts(
    n_generations: int = 1,
    batch_size: int = 2,
    max_new_tokens: int = 1024,
    num_threads: int = 4,
    model: str = "gpt-4",
):
    """Generates prompts for Pydantic classes. We want to generate prompts that return Pydantic definitions of software entities.
    For example:
    ```
    from pydantic import BaseModel, EmailStr

    class UserPersonalInfo(BaseModel):
        name: str
        email: EmailStr
        phone_number: str
        address: str
    ```
    Args:
        num_generations (int, optional): Number of generations to run. Defaults to 1.
        batch_size (int, optional): Batch size for generation. Defaults to 2.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
        num_threads (int, optional): Number of threads to use. Defaults to 4.
        model (str, optional): Model to use for generation. Defaults to "gpt-4".
    Returns:
        List[str]: List of generated prompts.
    """
    use_cases = [
        "User representation: A user profile is represented as a Pydantic class.",
        # "Company representation: A company profile is represented as a Pydantic class.",
        # "Message representation: A message is represented as a Pydantic class.",
        # "Booking representation: A booking is represented as a Pydantic class.",
        # "Novel representation: A novel is represented as a Pydantic class.",
        # "Author representation: An author is represented as a Pydantic class.",
        # "Software representation: A software application is represented as a Pydantic class.",
    ]
    use_case_dataset = Dataset.from_dict({"input": use_cases})
    generator_task = SelfInstructTask(
        application_description="Represents software application use cases as Pydantic classes.",
        system_prompt=(
            "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks."
            "You are given a task description and a set of instructions for how to write the prompts for a specific AI application."
            "You specialise in writing prompts for Pydantic classes."
            "You write prompts that require contained python code and not edit existing code."
            "You do not write prompts that require the user to explain or describe existing code."
        ),
    )
    instruction_generator = OpenAILLM(
        task=generator_task,
        num_threads=num_threads,
        max_new_tokens=max_new_tokens,
        model=model,
    )
    pipeline = Pipeline(generator=instruction_generator)
    distiset = pipeline.generate(
        dataset=use_case_dataset, num_generations=n_generations, batch_size=batch_size
    )
    # get a sample and log it clearly
    prompt_samples = [sample["generation_prompt"][0][1]["content"] for sample in distiset]
    input_samples = [sample["input"] for sample in distiset]
    log_input_generations(
        inputs=input_samples,
        generations=prompt_samples,
        message="Generated promps for the following use cases:",
    )
    return distiset

