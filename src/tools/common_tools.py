import os
import requests
import json
from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from src.llms.chatmodel import EinsteinChatModel
from src.gus.login import SfdcOrg
from typing import Optional
import json
import requests

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
#GUS WI TOOl#
class GusWorkItemInput(BaseModel):
    worksubject : str = Field(description="The subject of the GUS work item.",examples=["Hackathon 10.0"])
    workdetails: Optional[str] = Field(description="The detailed description of the GUS work item.",examples=["creating hack project"])
    useremail: Optional[str] = Field(description="Email Id of the user.",examples=["ankita.tiwari@salesforce.com"])
@tool("gus-wi-tool", args_schema=GusWorkItemInput, return_direct=False)
def create_story(worksubject: str, workdetails: str = None, useremail: str = None)-> str:
    """Use this tool with arguments like "{{"worksubject": str, "workdetails": str,"useremail": str }}" when you need to Create the GUS Work Item  based on the subject and description."""
    if not workdetails and not useremail:
        json_input = json.loads(worksubject)
        worksubject = json_input.get("worksubject")
        workdetails = json_input.get("workdetails")
        useremail = json_input.get("useremail")
    useremail = "ankita.tiwari@salesforce.com"
    if(os.getenv('GUS_DISABLE_WORK_ITEM',"true") == 'false'):
        try:
            # industriesObj=Industries()
            logIn=SfdcOrg(os.getenv('GUS_URL'),os.getenv('GUS_LOGIN_ID'),os.getenv('GUS_LOGIN_PWD'))
            authorization_header={'Authorization': 'Bearer '+ logIn.getSessionId(), 'content-type': 'application/json'}
            rest_url = "https://" + logIn.getServerHostname()+"/services/data/v50.0/query/?q=SELECT+Id+FROM+User+WHERE+email='"+useremail+"'"
            qa_response = requests.get(rest_url, headers=authorization_header, proxies=logIn.proxies)
            qa_response.raise_for_status()
            qa_user_id = qa_response.json()['records'][0]['Id']
            authorization_header = {'Authorization': 'Bearer '+logIn.getSessionId(), 'content-type': 'application/json'}
            # scrum_team = quote(os.getenv('GUS_SF_SCRUM_TEAM'))
            rest_url = (f"https://{logIn.getServerHostname()}/services/data/v50.0/query/"f"?q=SELECT+Id+FROM+agf__ADM_Sprint__c+WHERE+agf__Scrum_Team__c='{os.getenv('GUS_SF_SCRUM_TEAM_ID')}'"f"+AND+agf__Days_Remaining__c!='CLOSED'+AND+agf__Days_Remaining__c!='NOT STARTED'")
            sprint_response = requests.get(rest_url, headers=authorization_header, proxies=logIn.proxies)
            sprint_response.raise_for_status()
            sprint_id = sprint_response.json()['records'][0]['Id']

            mydata = {
                "agf__QA_Engineer__c": qa_response.json()['records'][0]['Id'],
                "agf__Scrum_Team__c": os.environ['GUS_SF_SCRUM_TEAM_ID'],
                "agf__Subject__c": worksubject,
                "RecordTypeId": "012Qy000006QqntIAC",
                "agf__Sprint__c": sprint_response.json()['records'][0]['Id'],
                "agf__Details__c": workdetails,
                "agf__Product_Tag__c": os.environ['GUS_SF_PRODUCT_TAG_ID'],
                "agf__Story_Points__c": 0
            }
            authorization_header = {'Authorization': 'Bearer ' +logIn.getSessionId(), 'content-type': 'application/json'}
            rest_url = "https://"+logIn.getServerHostname()+"/services/data/v50.0/sobjects/agf__ADM_Work__c"
            response = requests.post(rest_url, json=mydata, headers=authorization_header, proxies=logIn.proxies)
            response.raise_for_status()
            print(response.json())
            response_json = response.json()
            created_id = response_json.get("id")
            rest_url = f"https://{logIn.getServerHostname()}/services/data/v50.0/sobjects/agf__ADM_Work__c/{created_id}?fields=Name"
            get_response = requests.get(rest_url, headers=authorization_header, proxies=logIn.proxies)
            get_response.raise_for_status()
            work_item_id = get_response.json().get('Name')
            work_item_link = f"https://{logIn.getServerHostname()}/{created_id}"
            return json.dumps({
            "work_item_id": work_item_id,
            "link": work_item_link
        })
        except Exception as e:
            print("Error occurred while creating GUS work item:")
            print(e)
            if response is not None:
                print("Response status:", response.status_code)
                print("Response body:", response.text)  # <-- THIS LINE!
            return "null"
    else:
        print("GUS_DISABLE_WORK_ITEM set as true, hence skipped story creation") 
        return None   


class GithubIssuesToolInput(BaseModel):
    repo: str = Field(description="Name of the GitHub repository to fetch issues from", examples=["IPL-Score-Tracker", "Solution Cloud"])
    state: Optional[str] = Field(description="State of the issues to fetch. That is, open or closed", examples=["open", "closed"])
    
    
@tool("github-tool", args_schema=GithubIssuesToolInput, return_direct=False)
def fetch_github_issues(repo: str, state: str = None):
    """Fetches issues of specified state (open or closed) from the specified GitHub repository"""
    if not state:
        temp = json.loads(repo)
        repo_name = temp.get("repo")
        state = temp.get("state")
    else:
        repo_name = repo
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    url = f"https://api.github.com/repos/aditya-balachander/{repo_name}/issues"
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
    fetch_splunk_gack_logs,create_story
]

# --- Example of how to use the tools ---
if __name__ == "__main__":
    llm = EinsteinChatModel(api_key=os.getenv('EINSTEIN_API_KEY'), disable_streaming=True)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, COMMON_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=COMMON_TOOLS, verbose=True)
    # agent_executor.invoke({"input": "what is the weather in Los Angeles?"})
    agent_executor.invoke({"input": "Use the gus-wi-tool to create a work item. The subject is 'comparing two json input files'. The details are 'compare two json files and create one input json to fetch all license information'. The user's email is ankita.tiwari@salesforce.com."})
    agent_executor.invoke({"input": "Find out more information from Splunk about the gack with id 12345"})
