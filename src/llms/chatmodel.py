import json
import os
import logging
import time
import datetime
from typing import Any, Dict, List, Optional, Literal
import requests
from requests.exceptions import ConnectionError
from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, ChatMessage, ToolCall
) 
from langchain_core.messages.tool import ToolCall
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser, JsonOutputKeyToolsParser
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent

# --- Type Aliases ---
DictStrAny = Dict[str, Any]
load_dotenv()
# --- Logging Setup ---
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF_FACTOR = 1

class EinsteinChatModel(BaseChatOpenAI):
    # --- Configuration Parameters ---
    gateway_server: str = os.environ.get("EINSTEIN_GATEWAY_SERVER")
    gateway_path: str = "v1.0"
    api_key: str = os.environ.get("EINSTEIN_API_KEY")
    feature_id: str = "EinsteinDocsAnswers"
    app_context: str = "EinsteinGPT"
    core_tenant_id: str = "core/prod1/00DDu0000008cuqMAA"
    model_name: str = "llmgateway__OpenAIGPT4_32k"

    # --- Core Generation Parameters ---
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    num_generations: int = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # --- Einstein Specific Parameters ---
    composition_settings: Optional[DictStrAny] = None
    enable_pii_masking: Optional[bool] = None
    enable_input_safety_scoring: Optional[bool] = None
    enable_output_safety_scoring: Optional[bool] = True
    enable_input_bias_scoring: Optional[bool] = None
    enable_output_bias_scoring: Optional[bool] = None
    localization: Optional[DictStrAny] = None
    tags: Optional[Dict[str, str]] = None
    turn_id: Optional[str] = None
    slots_to_data: Optional[DictStrAny] = None
    debug_settings: Optional[DictStrAny] = None
    system_prompt_strategy: Optional[str] = None

    # --- Tool Calling Parameters ---
    tools: Optional[List[DictStrAny]] = None
    tool_config: Optional[DictStrAny] = None
    is_structured_output: bool = False

    # Internal client using requests
    _client = requests.Session()

    def _get_message_role(self, message: BaseMessage) -> str:
        """Maps LangChain message types to Einstein API role strings."""
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, ToolMessage):
            return "tool"
        elif isinstance(message, ChatMessage):
            return message.role
        else:
            logging.warning(f"Unknown message type {type(message)}, defaulting to 'user' role.")
            return "user"

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Format LangChain messages to the Einstein API /chat/generations format,
        including handling for AIMessage with tool_calls and ToolMessage results.
        """
        formatted = []
        for msg in messages:
            role = self._get_message_role(msg)
            content_val = msg.content

            message_dict: Dict[str, Any] = {"role": role}

            # --- Handle AIMessage with Tool Calls (Input History) ---
            # Check if it's an AIMessage AND has a non-empty tool_calls attribute
            msg_tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(msg, AIMessage) and msg_tool_calls:
                # Set content (might be empty or contain text preceding the call)
                message_dict["content"] = content_val if isinstance(content_val, str) else ""

                # --- Format Tool Calls for API Payload ---
                api_tool_calls = []
                for tc in msg_tool_calls:
                    if not isinstance(tc, dict) or not tc.get("id") or not tc.get("name") or tc.get("args") is None:
                        logging.warning(f"Skipping malformed tool_call object in AIMessage: {tc}")
                        continue

                    try:
                        args_str = json.dumps(tc.get("args", {}))
                        api_tool_calls.append(
                            {
                                "id": tc["id"],
                                "function": {
                                    "name": tc["name"],
                                    "arguments": args_str,
                                }
                            }
                        )
                    except (TypeError, json.JSONDecodeError) as e:
                        logging.error(f"Failed to format tool call arguments for API: {e}. ToolCall: {tc}")

                if api_tool_calls:
                    message_dict["tool_invocations"] = api_tool_calls
                else:
                    logging.warning(f"AIMessage had tool_calls attribute but none could be formatted: {msg_tool_calls}")
                    message_dict["content"] = content_val if isinstance(content_val, str) else ""

            # --- Handle Tool Message (Result) ---
            elif isinstance(msg, ToolMessage):
                message_dict["content"] = str(content_val)
                message_dict["tool_call_id"] = msg.tool_call_id

            # --- Handle Regular Content Messages (Human, System, plain AI) ---
            else:
                if isinstance(content_val, str):
                    message_dict["content"] = content_val
                elif content_val is not None:
                    logging.warning(f"Message content type {type(content_val)} not fully handled, using str().")
                    message_dict["content"] = str(content_val)

            # Filter out None content unless it's an assistant message potentially holding tool calls
            if message_dict.get("content") is None and not (role == "assistant" and message_dict.get("tool_invocations")):
                if "content" in message_dict:
                    del message_dict["content"]

            # Append the fully constructed message dictionary
            if message_dict.get("role"):
                formatted.append(message_dict)
            else:
                logging.warning(f"Skipping message without role: {msg}")

        return formatted

    def _prepare_payload(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, stream: bool = False) -> Dict[str, Any]:
        """Prepare the full payload for the /chat/generations or /chat/generations/stream endpoint."""
        # --- Generation Settings ---
        generation_settings: Dict[str, Any] = {"num_generations": self.num_generations}
        # Ensure n=1 for streaming if required by API
        if stream and self.num_generations != 1:
             logging.warning("Streaming typically supports num_generations=1. Forcing n=1 for stream request.")
             generation_settings["num_generations"] = 1

        if self.max_tokens is not None: generation_settings["max_tokens"] = self.max_tokens
        if self.temperature is not None: generation_settings["temperature"] = self.temperature
        effective_stop = stop if stop is not None else self.stop_sequences
        if effective_stop is not None: generation_settings["stop_sequences"] = effective_stop
        if self.frequency_penalty is not None: generation_settings["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None: generation_settings["presence_penalty"] = self.presence_penalty
        nested_params = {}
        if self.top_p is not None: nested_params["top_p"] = self.top_p
        if nested_params: generation_settings["parameters"] = nested_params

        # --- Main Payload ---
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "generation_settings": generation_settings,
        }
        if stream:
             payload["stream"] = True # Explicitly add stream flag if API requires it in payload

        # --- Add Optional Top-Level Settings (including Tools) ---
        optional_settings_map = {
            "composition_settings": self.composition_settings,
            "enable_pii_masking": self.enable_pii_masking,
            "enable_input_safety_scoring": self.enable_input_safety_scoring,
            "enable_output_safety_scoring": self.enable_output_safety_scoring,
            "enable_input_bias_scoring": self.enable_input_bias_scoring,
            "enable_output_bias_scoring": self.enable_output_bias_scoring,
            "localization": self.localization,
            "tags": self.tags,
            "turn_id": self.turn_id,
            "slots_to_data": self.slots_to_data,
            "debug_settings": self.debug_settings,
            "system_prompt_strategy": self.system_prompt_strategy,
            "tools": self.tools, # Add tools if provided (e.g., by with_structured_output)
            "tool_config": self.tool_config, # Add tool_config if provided
        }
        for key, value in optional_settings_map.items():
            if value is not None:
                payload[key] = value
        
        return payload

    def _build_headers(self) -> Dict[str, str]:
        """Constructs the request headers."""
        api_key_value = self.api_key
        headers = {
            "Authorization": f"API_KEY {api_key_value}",
            "Content-Type": "application/json",
            "Accept": "application/json", # Good practice
        }
        if self.feature_id: headers["x-client-feature-id"] = self.feature_id
        if self.app_context: headers["x-sfdc-app-context"] = self.app_context
        if self.core_tenant_id: headers["x-sfdc-core-tenant-id"] = self.core_tenant_id
        return headers

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make the synchronous HTTP POST request using requests."""
        api_url = f"https://{self.gateway_server}/{self.gateway_path}/chat/generations"
        headers = self._build_headers()
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    verify=False
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                logging.error(f"Attempt {attempt + 1}: Request timed out: {e}")
                last_exception = ConnectionError(f"API request timed out after {attempt + 1} attempt(s): {e}")
                break
            except requests.exceptions.SSLError as e:
                logging.error(f"Attempt {attempt + 1}: SSL Error: {e}. Check certificates.")
                last_exception = ConnectionError(f"SSL verification failed after {attempt + 1} attempt(s): {e}")
                break
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Attempt {attempt + 1}: Connection Error: {e}")
                last_exception = ConnectionError(f"Failed to connect to API after {attempt + 1} attempt(s): {e}")
                if attempt < MAX_RETRIES:
                     delay = INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                     logging.warning(f"Connection error on attempt {attempt + 1}. Retrying in {delay:.2f} seconds...")
                     time.sleep(delay)
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < MAX_RETRIES:
                    delay = INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logging.warning(
                        f"Attempt {attempt + 1}: Received HTTP 429 (Too Many Requests). "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    last_exception = e # Store the exception
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Attempt {attempt + 1}: HTTP Error {e.response.status_code}: {e.response.text}")
                    try: error_details = e.response.json()
                    except json.JSONDecodeError: error_details = e.response.text
                    last_exception = ValueError(f"API returned HTTP error: {e.response.status_code} - {error_details}")
                    break
            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {attempt + 1}: Request failed: {e}")
                last_exception = ConnectionError(f"Failed to make request to API after {attempt + 1} attempt(s): {e}")
                break
            except json.JSONDecodeError as e:
                resp_text = response.text if 'response' in locals() else 'No response text available'
                logging.error(f"Attempt {attempt + 1}: Failed to decode JSON response: {e}. Response text: {resp_text}")
                last_exception = ValueError(f"Invalid JSON received from API: {e}")
                break
        
        logging.error(f"Request failed after {attempt + 1} attempts.")
        if last_exception:
            raise last_exception
        else:
            raise ConnectionError("Request failed after multiple attempts for an unknown reason.")


    def _process_tool_calls(
            self, tool_calls_data: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """Processes raw tool call data from API into LangChain ToolCall objects."""
        tool_calls = []
        if not isinstance(tool_calls_data, list):
             logging.warning(f"Expected list for tool_calls_data, got {type(tool_calls_data)}. Skipping.")
             return tool_calls

        for tool_call_info in tool_calls_data:
             if not isinstance(tool_call_info, dict):
                 logging.warning(f"Skipping invalid tool call item (not a dict): {tool_call_info}")
                 continue

             # Assuming OpenAI structure - verify with Einstein API docs
             tool_id = tool_call_info.get("id")
             function_info = tool_call_info.get("function")

             if not tool_id or not isinstance(function_info, dict):
                 logging.warning(f"Skipping invalid tool call format: {tool_call_info}")
                 continue

             func_name = function_info.get("name")
             # Arguments are expected as a JSON *string*
             func_args_str = function_info.get("arguments")

             if not func_name or func_args_str is None: # Allow empty args string
                 logging.warning(f"Skipping tool call with missing name or arguments: {tool_call_info}")
                 continue

             try:
                 # Parse the arguments string into a dict
                 args_dict = json.loads(func_args_str)
                 tool_calls.append(
                     ToolCall(name=func_name, args=args_dict, id=tool_id)
                 )
             except json.JSONDecodeError as e:
                 logging.error(f"Failed to parse tool arguments JSON: {e}. Args string: '{func_args_str}'")
             except Exception as e:
                 logging.error(f"Error creating ToolCall object: {e}")

        return tool_calls

    def _process_response(self, response_data: Dict[str, Any]) -> ChatResult:
        """
        Process the response from the /chat/generations API, including tool calls,
        looking for tool data under the 'tool_invocations' key.
        """
        try:
            generation_details = response_data.get("generation_details")
            if not isinstance(generation_details, dict):
                if "error" in response_data: raise ValueError(f"API Error: {response_data['error']}")
                raise ValueError("Invalid response: 'generation_details' missing/invalid.")

            api_generations = generation_details.get("generations")
            if not isinstance(api_generations, list):
                raise ValueError("Invalid response: 'generation_details.generations' missing or not a list.")

            generations = []
            for gen_item in api_generations:
                if not isinstance(gen_item, dict):
                    logging.warning(f"Skipping invalid generation item: {gen_item}")
                    continue

                gen_params = gen_item.get("parameters", {})
                finish_reason = gen_params.get("finish_reason")
                # Get content; it might be None or empty string for tool calls
                content = gen_item.get("content")
                role = gen_item.get("role", "assistant") # Should be assistant

                ai_message: AIMessage
                parsed_tool_calls: List[ToolCall] = []

                # --- Check for Tool Calls ---
                if finish_reason == "tool_calls":
                    raw_tool_invocations = gen_item.get("tool_invocations")
                    if raw_tool_invocations:
                        parsed_tool_calls = self._process_tool_calls(raw_tool_invocations)
                    else:
                        logging.warning("Finish reason is 'tool_calls' but no 'tool_invocations' data found in generation item.")
                    
                    ai_message = AIMessage(content=content if isinstance(content, str) else "", tool_calls=parsed_tool_calls)
                else:
                    ai_message = AIMessage(content=content if isinstance(content, str) else "")

                # --- Gather Generation Info ---
                generation_info = {
                    "generation_id": gen_item.get("id"),
                    "timestamp": gen_item.get("timestamp"),
                    "finish_reason": finish_reason, # Store the actual finish reason
                    "index": gen_params.get("index"),
                    "logprobs": gen_params.get("logprobs"),
                    "generation_safety_score": gen_item.get("generation_safety_score"),
                    "generation_content_quality": gen_item.get("generation_content_quality"),
                    "raw_tool_invocations": gen_item.get("tool_invocations") if finish_reason == "tool_calls" else None
                }
                generation_info = {k: v for k, v in generation_info.items() if v is not None}

                generations.append(ChatGeneration(message=ai_message, generation_info=generation_info))

            # --- Overall LLM Output ---
            overall_params = generation_details.get("parameters", {})
            llm_output = {
                "transaction_id": response_data.get("id"),
                "model_name": overall_params.get("model"),
                "object_type": overall_params.get("object"),
                "usage": overall_params.get("usage"),
                "other_details": response_data.get("other_details"),
                "provider": overall_params.get("provider"),
                "system_fingerprint": overall_params.get("system_fingerprint"),
            }
            llm_output = {k: v for k, v in llm_output.items() if v is not None}

            if not generations and api_generations is not None:
                logging.warning("API returned an empty list of generations.")

            return ChatResult(generations=generations, llm_output=llm_output)

        except (AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logging.exception(f"Error processing API response: {e}")
            try: raw_response_str = json.dumps(response_data, indent=2)
            except Exception: raw_response_str = str(response_data)
            logging.error(f"Received response data: {raw_response_str}")
            raise ValueError(f"Failed to parse response from Einstein API: {e}") from e


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None, # Callback manager type hint
        **kwargs: Any,
    ) -> ChatResult:
        """ Main generation logic """
        # Allow overriding tools/tool_config via kwargs if needed
        current_tools = kwargs.get("tools", self.tools)
        current_tool_config = kwargs.get("tool_config", self.tool_config)

        # Temporarily update self.tools for _prepare_payload if overridden
        original_tools = self.tools
        original_tool_config = self.tool_config
        try:
            self.tools = current_tools
            self.tool_config = current_tool_config
            payload = self._prepare_payload(messages, stop=stop)
        finally:
            # Restore original values
            self.tools = original_tools
            self.tool_config = original_tool_config

        response_data = self._make_request(payload)
        return self._process_response(response_data)
    
    def with_structured_output(
            self,
            schema: Optional[Any] = None,
            *,
            method: Literal["function_calling", "json_mode"] = "function_calling",
            **kwargs: Any,
    ) -> RunnableSerializable[LanguageModelInput, Any]: # Return type adjusted
        """Enables structured output based on schema and method provided."""
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        if method not in ["function_calling", "json_mode"]:
            raise ValueError(f"Unrecognized method '{method}'. Must be 'function_calling' or 'json_mode'.")

        # --- Prepare Tools and Parser ---
        bound_tools: Optional[List[Dict[str, Any]]] = None
        output_parser: Optional[RunnableSerializable] = None
        is_pydantic_schema = is_basemodel_subclass(schema) if schema else False

        if method == "function_calling":
            if schema is None:
                raise ValueError("Schema must be provided for method 'function_calling'")
            tool_dict = convert_to_openai_tool(schema)
            bound_tools = [tool_dict]
            tool_name = tool_dict["function"]["name"]

            if is_pydantic_schema:
                 if not isinstance(schema, type): schema = type(schema)
                 output_parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                 output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)

        elif method == "json_mode":
            logging.info("Using 'json_mode'. Ensure the model/API supports it or use appropriate prompting.")
            if schema:
                 if is_pydantic_schema and not isinstance(schema, type): schema = type(schema)
                 output_parser = PydanticOutputParser(pydantic_object=schema) if is_pydantic_schema else JsonOutputParser()
            else:
                 output_parser = JsonOutputParser()

        # --- Create Bound Model ---
        model_with_tools = self.bind(tools=bound_tools) # Use bind method

        # --- Return Chain ---
        if output_parser:
            return model_with_tools | output_parser
        else:
            return model_with_tools

    @property
    def _llm_type(self) -> str:
        """Returns the type of this model."""
        return "einstein-llm-gateway-chat-tools" # Indicate tool support

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Identifying parameters."""
        params = super()._identifying_params
        return params


def init_agent(tools=None):
    """Initialize the LLM with the API key."""
    einstein_api_key = os.getenv('EINSTEIN_API_KEY')
    llm = EinsteinChatModel(api_key=einstein_api_key, disable_streaming=True)
    
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = f"""
    You are a highly capable AI assistant operating within a Slack workspace. Your purpose is to be helpful, accurate, and efficient in responding to user queries, formatting your responses strictly according to Slack standards.

    Today's date is {current_date}. Use this information if relevant to the user's query (e.g., interpreting 'last week').

    **Conversation Context:**
    1.  **Chat History (`chat_history`):** Use the provided recent conversation history from the current thread for context.
    2.  **User Input (`input` & `channel_id`):** The user's latest message is in `input`. The `channel_id` of the current Slack channel is also provided.

    **Available Tools:**
    You have access to the following tools:

    * **[Your Tool 1 Name] (`[your_tool_1_code]`):** [Description and when to use].
    * **[Your Tool 2 Name] (`[your_tool_2_code]`):** [Description and when to use].
    * **Slack Channel History Search (`slack_channel_history_search`):** Searches message history *of the current Slack channel*. Use when asked about past conversations or mentions *in this channel*. Requires `query` and `channel_id`. This tool will provide results including Slack message permalinks in the correct `<url|text>` format.
    * **Confluence Document Search (`confluence_document_search`):** Searches the indexed Confluence documentation. Use this when the user asks about procedures, official documentation, knowledge base articles, or 'how-to' questions likely answered in Confluence. Requires only the `query`. This tool will provide results including links to the Confluence pages in the correct `<url|text>` format.

    **Tool Usage Instructions:**
    1.  Analyze the query (`input`) and `chat_history`.
    2.  Determine if a tool is needed or if the answer is known/in history.
    3.  Select the *most appropriate* tool.
    4.  Provide required parameters accurately.
    5.  **CRITICAL for `slack_channel_history_search`:** MUST use the `channel_id` provided with the current `input`. Extract it and pass it as the `channel_id` parameter.
    6.  **For `confluence_document_search`:** Only provide the `query` parameter.

    **Response Guidelines:**
    * **Formatting:** Structure your response for optimal readability within Slack. Use standard Slack markdown:
        * `*bold*` for emphasis.
        * `_italic_` for less emphasis.
        * `~strike~` for strikethrough.
        * ` ```code block``` for multi-line code or logs.
        * `>` for blockquotes.
        * Use bullet points (`*` or `-`) for lists.
    * **Links:** **CRITICAL:** ALL links in your response MUST use Slack's link format: `<URL|Link Text>`. NEVER use Markdown link format like `[Link Text](URL)`.
    * **Synthesize Information:** Respond directly to the user's query based on your knowledge, the chat history, or the results from the tools. Clearly synthesize information retrieved from tools. Don't just output raw tool data unless specifically asked or appropriate (like logs).
    * **User-Friendly Time:** Avoid raw timestamps (like `1745507846.712679`). Refer to times relatively (e.g., "earlier today", "yesterday", "last week") or use standard date/time formats if specific dates are needed.
    * **Reference Links:** When citing information obtained from tools (Slack messages or Confluence documents), use reference-style links.
        1.  In the main body of your response where you mention the information, add a reference marker like `[1]`, `[2]`, etc.
        2.  At the end of your *entire* response, create a "References:" section.
        3.  List each source with its corresponding number and the *exact* Slack-formatted link (`<url|text>`) provided by the tool. Do NOT add extra text around the link in the reference list item itself.
        *Example:*
        ```
        Based on the discussion earlier today, the deployment was successful [1]. The standard procedure is also documented in Confluence [2].

        References:
        [1] <slack_permalink_url|Slack Message from User Name at Time>
        [2] <confluence_page_url|Confluence Page Title>
        ```
    * **Tool Errors:** If a tool fails or returns an error, inform the user clearly that you couldn't retrieve the information using that method.
    * **Unknown Answers:** If you cannot answer the question using your knowledge or the available tools, clearly state that.
    * **Tone:** Maintain a helpful and conversational tone.
    * **Conciseness:** Be concise but complete.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True), # Make history optional if it can be empty
            ("human", "Channel ID: {channel_id}\nUser Query: {input}"), # Combine channel_id and user input here
            MessagesPlaceholder(variable_name="agent_scratchpad"), # For tool calls/outputs
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor