from textwrap import dedent
from typing import Any


from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt
from distilabel.tasks.preference.ultrafeedback import Rating, UltraFeedbackTask
from dotenv import load_dotenv
from torch import batch_norm

load_dotenv("../.env")


class FunctionFeedbackTask(UltraFeedbackTask):
    @property
    def output_args_names(self):
        return ["rating", "feedback"]

    @property
    def input_args_names(self):
        return ["function", "instruction", "function_call"]

    def generate_prompt(
        self, function: str, instruction: str, function_call: str, **_: Any
    ) -> Prompt:
        input = f"{self.task_description}\n\n"
        input += f"Function: {function}\n\n"
        input += f"Instruction: {instruction}\n\n"
        input += "Ratings: 1-3\n\n"
        for rating in self.ratings:
            input += f"\n{rating['value']}: {rating['description']}"
        input += f"\n\nResponse: {function_call}"
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input,
        )

    def parse_output(self, output: str) -> Any:
        feedback_rating = output.split(": ")
        try:
            rating, feedback = feedback_rating
        except ValueError:
            if isinstance(feedback_rating, list):
                feedback_rating = feedback_rating[0]
            feedback_rating = feedback_rating.strip()
            rating = feedback_rating if feedback_rating.isnumeric() else 0
            feedback = "No feedback"
        except Exception as e:
            print(e)
            rating = 0
            feedback = "No feedback: Could not parse output."
        output = {
            "feedback": feedback,
            "rating": int(rating),
        }
        return output


def generate_feedback(
    dataset: "Dataset",
    batch_size: int = 5,
    num_generations: int = 2,
) -> "CustomDataset":
    ultrafeedback_task = FunctionFeedbackTask(
        system_prompt="Your role is to evaluate text quality based on given criteria",
        task_description=dedent(
            """
            # JSON Schema Validity Assessment
            Evaluate the model's json schema based on various criteria:
            1. **Correctness**: Does the output provide accurate and relevant examples within the JSON fields?
            2. **Instruction Following**: Does the JSON align with given instructions and the user's intent?
            3. **Completeness**: Does the JSON schema represent the instruction fully?

            **Scoring**: Rate outputs 1 to 3 based on the overall quality, considering all aspects:
            """
        ),
        ratings=[
            Rating(
                value=1,
                description="The JSON schema is incomplete and does not represent the instruction.",
            ),
            Rating(
                value=2,
                description="The JSON schema is complete but field, descriptions, and examples should be improved.",
            ),
            Rating(
                value=3,
                description="The JSON schema is complete and represents the instruction fully.",
            ),
        ],
    )
    labeller = OpenAILLM(
        task=ultrafeedback_task,
        max_new_tokens=2048,
        model="gpt-4",
    )
    pipeline = Pipeline(labeller=labeller)
    feedback_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=None,
    )
    return feedback_dataset
