from typing import Any, Dict
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import pprint
import os
from dotenv import load_dotenv, find_dotenv
from helpers import printObject
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

_ = load_dotenv(find_dotenv())  # read local .env file


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


# Prompt ------------------------------------------------
# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
printObject("prompt", prompt)


# Model ------------------------------------------------
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
printObject("llm", llm)

# Agent ------------------------------------------------

converted_tools = []
for tool in tools:
    printObject(f"tool {tool}", tool)
    converted = convert_to_openai_tool(tool)
    print(f"converted {tool}", converted)
    converted_tools.append(converted)

llm_with_tools = llm.bind(tools=converted_tools)
printObject("llm with tools", llm_with_tools)

agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        )
    )
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
printObject("agent", agent)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})
