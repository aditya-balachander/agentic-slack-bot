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


class SplunkGackInput(BaseModel):
    gack_id: str = Field(description="The Gack ID to fetch logs for", examples=["GACK-12345", "GACK-98765"])

@tool("splunk-gack-logs", args_schema=SplunkGackInput, return_direct=False)
def fetch_splunk_gack_logs(gack_id: str) -> str:
    """Fetch logs for a Gack ID from Splunk (mocked response)."""
    # Sample logs - in reality you'd fetch from Splunk
    sample_logs = f"""
    [INFO] 2025-04-25 12:32:10,321 [req-78b1] Starting request for Gack ID {gack_id}
    [DEBUG] 2025-04-25 12:32:10,322 [req-78b1] Request metadata: {{ "user": "mjawadtp", "org": "00Dxx0000001gYk", "action": "ProcessCase" }}

    [ERROR] 2025-04-25 12:32:12,104 [req-78b1] java.lang.RuntimeException: Unexpected server error while processing request
        at com.salesforce.core.workflow.Engine.execute(Engine.java:174)
        at com.salesforce.core.workflow.Engine.run(Engine.java:102)
        at com.salesforce.api.controller.GackController.handle(GackController.java:58)
        at com.salesforce.api.controller.GackController$$FastClassBySpringCGLIB$$e2f4f3d4.invoke(<generated>)
        at org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)
        at org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.invokeJoinpoint(CglibAopProxy.java:793)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)
        at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:123)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
        at org.springframework.aop.framework.CglibAopProxy$DynamicAdvisedInterceptor.intercept(CglibAopProxy.java:688)
    Caused by: java.lang.NullPointerException: Cannot invoke "String.trim()" because "inputData" is null
        at com.salesforce.data.transform.TransformService.process(TransformService.java:98)
        at com.salesforce.data.transform.TransformService.transform(TransformService.java:61)
        at com.salesforce.engine.steps.TransformStep.execute(TransformStep.java:34)
        at com.salesforce.engine.workflow.StandardWorkflow.executeStep(StandardWorkflow.java:89)
        at com.salesforce.engine.workflow.StandardWorkflow.run(StandardWorkflow.java:45)
        at com.salesforce.core.workflow.Engine.execute(Engine.java:172)
        ... 10 more

    [DEBUG] 2025-04-25 12:32:13,110 [req-78b1] Retrying workflow for Gack ID {gack_id}, attempt 2
    [ERROR] 2025-04-25 12:32:15,882 [req-78b1] TimeoutException: External API call exceeded timeout of 3000ms
        at com.salesforce.external.ExternalServiceClient.call(ExternalServiceClient.java:54)
        at com.salesforce.engine.steps.ExternalAPIStep.execute(ExternalAPIStep.java:29)
        at com.salesforce.engine.workflow.StandardWorkflow.executeStep(StandardWorkflow.java:93)

    [INFO] 2025-04-25 12:32:18,001 [req-78b1] Gack ID {gack_id} logged and escalated to engineering
    """
    return sample_logs.strip()


COMMON_TOOLS = [
    weather,
    fetch_github_issues,
    fetch_splunk_gack_logs  # ← Add this line
]

# --- Example of how to use the tools ---
if __name__ == "__main__":
    llm = EinsteinChatModel(api_key=os.getenv('EINSTEIN_API_KEY'), disable_streaming=True)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, COMMON_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=COMMON_TOOLS, verbose=True)
    agent_executor.invoke({"input": "Find out more information from Splunk about the gack with id 12345"})