import os
import requests
import json
from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from src.llms.chatmodel import EinsteinChatModel

# --- Define Tools ---

class WeatherToolInput(BaseModel):
    city: str = Field(description="Name of the city to get weather for.", examples=["New York", "Los Angeles"])

@tool("weather-tool", args_schema=WeatherToolInput, return_direct=False)
def weather(city: str) -> str:
    """Get the weather for a given city."""
    if city.lower() == "new york":
        return "It's sunny in New York!"
    elif city.lower() == "los angeles":
        return "It's rainy in Los Angeles!"
    else:
        return f"Sorry, I don't have weather information for {city}."


class GithubIssuesToolInput(BaseModel):
    repo: str = Field(description="Name of the GitHub repository to fetch issues from", examples=["IPL-Score-Tracker", "Solution Cloud"])
    state: Optional[str] = Field(description="State of the issues to fetch. That is, open or closed", examples=["open", "closed"])
    
    
@tool("github-tool", args_schema=GithubIssuesToolInput, return_direct=False)
def fetch_github_issues(repo: str):
    """Fetches issues of specified state (open or closed) from the specified GitHub repository"""
    temp = json.loads(repo)
    repo_name = temp.get("repo")
    state = temp.get("state")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    url = f"https://api.github.com/repos/mjawadtp/{repo_name}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "state": state  # 'open', 'closed', or 'all'
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch issues: {response.status_code} - {response.text}")


# --- Add all tools here ---
COMMON_TOOLS = [
    weather,
    fetch_github_issues
]

# --- Example of how to use the tools ---
if __name__ == "__main__":
    llm = EinsteinChatModel(api_key=os.getenv('EINSTEIN_API_KEY'), disable_streaming=True)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, COMMON_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=COMMON_TOOLS, verbose=True)
    agent_executor.invoke({"input": "Can you fetch open issues from the repo IPL-Score-Tracker and list them along with their description"})