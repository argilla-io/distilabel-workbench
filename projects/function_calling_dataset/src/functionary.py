import json

from distilabel.llm import OpenAILLM
from distilabel.llm.utils import LLMOutput
from distilabel.tasks import TextGenerationTask
from openai import OpenAI


functionary_client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")


class FunctionResponseGeneratorTask(TextGenerationTask):

    @property
    def input_args_names(self):
        return ["instructions", "function"]

    @property
    def output_args_names(self):
        return ["function_call"]

    def parse_output(self, output: str):
        function_name = output[0].function.name
        function_arguments = output[0].function.arguments
        if isinstance(function_arguments, str):
            function_arguments = json.loads(function_arguments)
        function = {"name": function_name, "arguments": function_arguments}
        return {"function_call" : json.dumps({"function_calls" : [function]})}

    def generate_prompt(self, instruction: str, function: str, **_):
        return instruction, function



class FunctionUseOpenAILLM(OpenAILLM):

    def _make_messages(self, instruction):
        return [
            {"role": "user", "content": instruction}
        ]

    def _load_function(self, inputs):
        return json.loads(inputs["function"].replace("```json","").replace("```",""))

    def _generate(
        self,
        inputs,
        num_generations,
    ):
        outputs = []
        for input in inputs:
            function = self._load_function(input)
            output = []
            for instruction in input["instructions"]:
                messages = self._make_messages(instruction)
                try:
                    chat_completions = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model,
                        n=num_generations,
                        max_tokens=self.max_tokens,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=50,
                        tools = [function],
                        tool_choice="auto"
                    )
                    chat_completion = chat_completions.choices[0]
                    parsed_response = self.task.parse_output(
                        chat_completion.message.tool_calls
                    )
                except Exception as e:
                    print(f"Error calling Functionary Server: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=messages,
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs

# functionary_llm_small = FunctionUseOpenAILLM(client=functionary_client, model="meetkai/functionary-small-v2.2", task=FunctionResponseGeneratorTask())
functionary_llm_medium = FunctionUseOpenAILLM(client=functionary_client, model="meetkai/functionary-medium-v2.2", task=FunctionResponseGeneratorTask())