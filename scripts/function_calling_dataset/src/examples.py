from typing import ClassVar, Dict, List

from pydantic import BaseModel


####################
# Example Functions
####################


class Properties(BaseModel):
    type: str
    description: str
    items: Dict[str, str] | None = None


class Parameters(BaseModel):
    type: ClassVar[str] = "object"
    properties: Dict[str, Properties]
    required: List[str]


class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters


class Tool(BaseModel):
    type: ClassVar[str] = "function"
    function: Function


bing_tool = Tool(
    type="function",
    function=Function(
        name="bing",
        description="This function returns the first 10 search results for a given query.",
        parameters=Parameters(
            type="object",
            properties={
                "query": Properties(type="string", description="The search query.")
            },
            required=["query"],
        ),
    ),
)

weather_tool = Tool(
    type="function",
    function=Function(
        name="weather",
        description="This function returns the current weather for a given location.",
        parameters=Parameters(
            type="object",
            properties={
                "location": Properties(
                    type="string", description="The location to get the weather for."
                )
            },
            required=["location"],
        ),
    ),
)


delete_file_tool = Tool(
    type="function",
    function=Function(
        name="delete_file",
        description="This function deletes a file from the file system.",
        parameters=Parameters(
            type="object",
            properties={
                "file_path": Properties(
                    type="string", description="The path to the file to delete."
                )
            },
            required=["file_path"],
        ),
    ),
)

example_tools = [bing_tool, weather_tool, delete_file_tool]

example_function_domain = [
    "Functions for search APIs",
    "Functions for weather APIs",
    "Functions for web scraping",
    "Functions for file manipulation",
    "Functions for developer tools",
    "Functions for chat management",
    "Functions for email management",
    "Functions for calendar management",
    "Functions for note taking",
    "Functions for task management",
    "Functions for project management",
    "Functions for stock market APIs",
    "Functions for news APIs",
    "Functions for sports APIs",
    "Functions for social media APIs",
    "Functions for travel APIs",
    "Functions for food delivery APIs",
    "Functions for data visualization",
    "Functions for data analysis",
]


####################
# Example Responses
####################


class FunctionCallResponse(BaseModel):
    name: str
    arguments: Dict[str, str]


class FunctionCallResponseArray(BaseModel):
    function_calls: List[FunctionCallResponse]


example_function_responses = FunctionCallResponseArray(
    function_calls=[
        FunctionCallResponse(
            name="search_bing", arguments={"query": "Apple stock price"}
        ),
        FunctionCallResponse(
            name="move_file",
            arguments={"source": "file1.txt", "destination": "file2.txt"},
        ),
        FunctionCallResponse(
            name="get_weather", arguments={"location": "New York, NY"}
        ),
    ]
)
