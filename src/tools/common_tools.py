import os

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from src.llms.chatmodel import EinsteinChatModel
from src.gus.login import SfdcOrg
from typing import Optional
import json
import requests
# from urllib.parse import quote

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
def create_story(json_input: str)-> str:
    """Use this tool with arguments like "{{"worksubject": str, "workdetails": str,"useremail": str }}" when you need to Create the GUS Work Item  based on the subject and description."""
    json_input = json.loads(json_input)
    worksubject = json_input.get("worksubject")
    workdetails = json_input.get("workdetails")
    useremail = json_input.get("useremail")
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
            newdata = str(mydata).replace("'", '"')
            authorization_header = {'Authorization': 'Bearer ' +logIn.getSessionId(), 'content-type': 'application/json'}
            rest_url = "https://"+logIn.getServerHostname()+"/services/data/v50.0/sobjects/agf__ADM_Work__c"
            print("Payload being sent:")
            print(newdata)
            response = requests.post(rest_url, data=newdata, headers=authorization_header, proxies=logIn.proxies)
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

# --- Add all tools here ---
COMMON_TOOLS = [
    weather,create_story
]

# --- Example of how to use the tools ---
if __name__ == "__main__":
    llm = EinsteinChatModel(api_key=os.getenv('EINSTEIN_API_KEY'), disable_streaming=True)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, COMMON_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=COMMON_TOOLS, verbose=True)
    # agent_executor.invoke({"input": "what is the weather in Los Angeles?"})
    agent_executor.invoke({"input": "Use the gus-wi-tool to create a work item. The subject is 'comparing two json input files'. The details are 'compare two json files and create one input json to fetch all license information'. The user's email is ankita.tiwari@salesforce.com."})