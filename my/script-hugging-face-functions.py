import os
import sys
from typing import Any, Dict

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers.string import StrOutputParser

# sys.path.append("./")

from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from helpers import printObject
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.chat_models.huggingface_functions import (
    ChatHuggingFaceFunctions,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv, find_dotenv
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.llms import HuggingFaceEndpoint


_ = load_dotenv(find_dotenv())  # read local .env file
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    endpoint_url="https://ax17kk73vpkws0jx.us-east-1.aws.endpoints.huggingface.cloud",
    task="text-generation",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_token,
)


# Tools
# tools = [TavilySearchResults(max_results=1)]

# tools = [
#     {
#         "name": "tavily_search_results_json",
#         "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {"description": "search query to look up", "type": "string"}
#             },
#             "required": ["query"],
#         },
#     }
# ]


@tool
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather in a given location.

    Parameters:
    - location (str): The city, e.g., 'San Francisco'.
    - unit (str): The temperature unit to use, 'celsius' or 'fahrenheit'. Defaults to 'celsius'.

    Returns:
    - dict: A dictionary containing weather data.
    """
    # Weather data for specific cities
    weather_data = {
        "London": {
            "celsius": {"temperature": "15 C", "condition": "Cloudy"},
            "fahrenheit": {"temperature": "59 F", "condition": "Cloudy"},
        },
        "Dublin": {
            "celsius": {"temperature": "18 C", "condition": "Partly Cloudy"},
            "fahrenheit": {"temperature": "64 F", "condition": "Partly Cloudy"},
        },
    }

    # Check if weather data exists for the given location and unit
    if location in weather_data and unit in weather_data[location]:
        return weather_data[location][unit]
    else:
        return {"message": "No data available for the specified location."}


@tool
def get_clothes(temperature: str, condition: str) -> str:
    """
    Function to recommend clothing based on temperature and weather condition.

    Parameters:
    - temperature (str): The temperature, e.g., '60 F' or '15 C'.
    - condition (str): The weather condition, e.g., 'Sunny', 'Rainy'.

    Returns:
    - str: A string suggesting appropriate clothing for the given weather, or an error message.
    """

    # Validate temperature input
    if not isinstance(temperature, str) or len(temperature.split()) != 2:
        return "Invalid temperature input. Please provide a temperature in format XX F or XX C."

    temp, unit = temperature.split()
    unit = unit.upper()  # Normalize unit to uppercase
    if unit not in ["F", "C"]:
        return "Invalid temperature unit. Please use F for Fahrenheit or C for Celsius."

    try:
        temp_value = int(temp)
        if unit == "C":
            # Convert Celsius to Fahrenheit for consistency
            temp_value = temp_value * 9 / 5 + 32
    except ValueError:
        return "Invalid temperature value. Please provide a numeric temperature."

    # Normalize condition to capitalize case
    condition = (
        condition.capitalize()
    )  # This makes the first letter capital and the rest lower
    valid_conditions = [
        "Partly cloudy",
        "Cloudy",
        "Sunny",
        "Rainy",
        "Snowy",
        "Windy",
        "Foggy",
    ]
    if condition not in valid_conditions:
        return "Invalid condition input. Please provide a valid weather condition."

    # Recommendations based on temperature
    if temp_value >= 77:
        outfit = "light clothing, such as a t-shirt and shorts"
    elif 59 <= temp_value < 77:
        outfit = "moderate clothing, like a long-sleeve shirt and jeans"
    else:
        outfit = "warm clothing, including a sweater or coat"

    # Additional recommendations based on condition
    if condition == "Rainy":
        outfit += ", and don't forget an umbrella or a raincoat"
    elif condition == "Snowy":
        outfit += ", with a heavy coat, gloves, and a warm hat"
    elif condition == "Windy":
        outfit += ", and consider a windbreaker or jacket"
    elif condition == "Foggy":
        outfit += ", and a light jacket might be useful"

    return outfit


tools = [get_current_weather, get_clothes]


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location.\n\n    Parameters:\n    - location (str): The city, e.g., 'San Francisco'.\n    - unit (str): The temperature unit to use, 'celsius' or 'fahrenheit'. Defaults to 'celsius'.\n\n    Returns:\n    - dict: A dictionary containing weather data.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {"type": "string"},
#                     "unit": {"type": "string", "default": "celsius"},
#                 },
#                 "required": ["location"],
#             },
#             "returns": "any",
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_clothes",
#             "description": "Function to recommend clothing based on temperature and weather condition.\n\n    Parameters:\n    - temperature (str): The temperature, e.g., '60 F' or '15 C'.\n    - condition (str): The weather condition, e.g., 'Sunny', 'Rainy'.\n\n    Returns:\n    - str: A string suggesting appropriate clothing for the given weather, or an error message.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "temperature": {"type": "string"},
#                     "condition": {"type": "string"},
#                 },
#                 "required": ["temperature", "condition"],
#             },
#             "returns": "string",
#         },
#     },
# ]


converted_tools = []
for tool in tools:
    printObject(f"tool {tool}", tool)
    converted = convert_to_openai_function(tool)
    print(f"converted {tool}", converted)
    converted_tools.append(converted)


# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        BaseMessage(type="function_metadata", content=converted_tools),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
printObject("prompt", prompt)


# model
model = ChatHuggingFaceFunctions(
    llm=llm,
    model_id="Trelis/Mistral-7B-Instruct-v0.2-function-calling-v3",
)
printObject("llm", model)


# Agent ------------------------------------------------


model_with_tools = model.bind(functions=converted_tools)
printObject("llm with tools", model_with_tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | model_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "I am in London, what should I wear?"})
