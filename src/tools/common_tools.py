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
    """Input schema for the Splunk Gack Log fetcher tool."""
    gack_id: str = Field(
        description="The Gack ID to fetch logs for.",
        examples=["g-12345", "g-98765"] # Adjusted example format
    )
    earliest_time: str = Field(
        default="-3h",
        description="The earliest time for the search window (Splunk relative time format). Defaults to 3 hours ago.",
        examples=["-1h", "-15m", "-1d"]
    )
    latest_time: str = Field(
        default="now",
        description="The latest time for the search window (Splunk relative time format). Defaults to now.",
        examples=["now", "-5m"]
    )
    max_results: int = Field(
        default=100,
        description="Maximum number of log events to return.",
        examples=[50, 200]
    )

# --- Tool Implementation ---
@tool("splunk-gack-logs", args_schema=SplunkGackInput, return_direct=False)
def fetch_splunk_gack_logs(gack_id: str, earliest_time: str = "-3h", latest_time: str = "now", max_results: int = 100) -> str:
    """
    Fetches logs for a specific Gack ID from Splunk within a given time range.
    Requires SPLUNK_HOST, SPLUNK_PORT, SPLUNK_USERNAME, SPLUNK_PASSWORD
    environment variables to be set. Optionally uses SPLUNK_INDEX and SPLUNK_SOURCETYPE.
    """
    logger.info(f"Attempting to fetch Splunk logs for Gack ID: {gack_id}, Time: {earliest_time} to {latest_time}")

    # --- Load Configuration ---
    splunk_host = os.getenv("SPLUNK_HOST")
    splunk_port = os.getenv("SPLUNK_PORT")
    splunk_user = os.getenv("SPLUNK_USERNAME")
    splunk_pass = os.getenv("SPLUNK_PASSWORD") # Consider using tokens if possible

    if not all([splunk_host, splunk_port, splunk_user, splunk_pass]):
        error_msg = "Splunk connection details (HOST, PORT, USERNAME, PASSWORD) missing in environment variables."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    # Optional index and sourcetype for more targeted search
    splunk_index = os.getenv("SPLUNK_INDEX", "*") # Default to all indexes if not set
    splunk_sourcetype = os.getenv("SPLUNK_SOURCETYPE", "*") # Default to all sourcetypes if not set

    # --- Connect to Splunk ---
    try:
        service = client.connect(
            host=splunk_host,
            port=splunk_port,
            username=splunk_user,
            password=splunk_pass
            # Add other connection args if needed (e.g., scheme='https', verify=False for self-signed certs - insecure!)
        )
        logger.info(f"Successfully connected to Splunk service at {splunk_host}:{splunk_port}")
    except Exception as e:
        error_msg = f"Failed to connect to Splunk: {e}"
        logger.exception(error_msg) # Log full traceback
        return f"Error: {error_msg}"

    # --- Construct and Execute Search Query ---
    # Basic query structure - adjust index, sourcetype, and fields as needed
    # Using f-string requires careful handling of quotes if gack_id could contain them
    # Ensure gack_id is properly escaped if necessary, though usually safe for typical IDs
    search_query = f'search index="{splunk_index}" sourcetype="{splunk_sourcetype}" "{gack_id}"'
    # Add time constraints and limit results
    search_query += f' | head {max_results}' # Limit results early
    # Add fields to display if needed, _raw shows the full event
    # search_query += ' | table _time, _raw, host, source'

    kwargs_search = {
        "earliest_time": earliest_time,
        "latest_time": latest_time,
        "exec_mode": "blocking" # Run the search synchronously
    }

    try:
        logger.info(f"Executing Splunk search: {search_query} with time range {earliest_time} to {latest_time}")
        # Start the job
        job = service.jobs.create(search_query, **kwargs_search)
        logger.info(f"Splunk search job created (SID: {job.sid}). Waiting for results...")

        # Wait for the job to finish (already done with blocking mode)
        # You could add a loop with status checks for normal mode

        # --- Process Results ---
        result_count = int(job["resultCount"])
        logger.info(f"Search completed. Found {result_count} results.")

        if result_count == 0:
            return f"No Splunk logs found for Gack ID '{gack_id}' between {earliest_time} and {latest_time}."

        log_entries = []
        # Get results using the results reader
        # Limiting here again just in case head didn't work as expected or for large result sets
        rr = results.ResultsReader(job.results(count=max_results))
        for result in rr:
            if isinstance(result, results.Message):
                # Log Splunk messages (e.g., warnings)
                logger.warning(f"Splunk Message: {result.type} {result.message}")
            elif isinstance(result, dict):
                # Extract the raw log event
                raw_log = result.get("_raw", "Log format does not contain _raw field.")
                log_entries.append(raw_log)

        if not log_entries:
             return f"Found {result_count} results, but could not extract log content."

        # Format the output string
        output = f"Found {len(log_entries)} log events for Gack ID '{gack_id}' ({earliest_time} to {latest_time}):\n\n"
        output += "\n".join(log_entries)
        return output.strip()

    except Exception as e:
        error_msg = f"An error occurred during Splunk search or processing: {e}"
        logger.exception(error_msg)
        # Try to cancel the job if it exists
        if 'job' in locals() and job:
            try:
                job.cancel()
                logger.info(f"Cancelled Splunk job {job.sid}")
            except Exception as cancel_e:
                logger.error(f"Failed to cancel Splunk job {job.sid}: {cancel_e}")
        return f"Error: {error_msg}"
    finally:
        # Ensure job is cancelled if it was created and might still be running
        if 'job' in locals() and job and not job.is_done():
             try:
                  job.cancel()
                  logger.info(f"Ensured cancellation of Splunk job {job.sid} in finally block.")
             except Exception:
                  pass # Ignore errors during final cleanup cancellation

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
