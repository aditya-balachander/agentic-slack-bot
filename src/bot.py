import os
import re
import logging
import atexit
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, AIMessage
from slack_sdk.errors import SlackApiError

from src.tools.common_tools import COMMON_TOOLS
from src.llms.chatmodel import init_agent
from src.tools.vector_store_manager import VectorStoreManager
from src.tools.slack_search_tool import SlackChannelHistorySearchTool
from src.tools.slack_loader import _create_document_from_slack_message
from src.llms.embeddings import EinsteinEmbeddings, MODEL
from src.tools.confluence_search_tool import ConfluenceSearchTool

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Confluence Config
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_PAGE_URLS_STR = os.getenv("CONFLUENCE_PAGE_URLS")
confluence_urls = []
if CONFLUENCE_PAGE_URLS_STR:
    confluence_urls = [url.strip() for url in CONFLUENCE_PAGE_URLS_STR.split(',')]
    logger.info(f"Loaded {len(confluence_urls)} Confluence URLs from environment.")
else:
    logger.warning("CONFLUENCE_PAGE_URLS environment variable not set or empty.")

# Initialize Bolt app
app = App(token=SLACK_BOT_TOKEN)

# --- Get Bot User ID ---
try:
    auth_response = app.client.auth_test()
    bot_user_id = auth_response['user_id']
    print(f"Detected Bot User ID: {bot_user_id}")
except Exception as e:
    print(f"Error fetching bot user ID: {e}. History reconstruction might be inaccurate.")
    bot_user_id = None # Handle case where auth_test fails

try:
    # Example using OpenAI embeddings
    embeddings = EinsteinEmbeddings(model=MODEL)
    # Test embedding
    _ = embeddings.embed_query("Test query")
    logger.info("Embeddings model initialized successfully.")
except Exception as e:
    logger.exception(f"Failed to initialize embeddings model: {e}. Exiting.")
    exit()

# --- Initialize Vector Store Manager ---
vector_store_manager = VectorStoreManager(
    embeddings=embeddings,
    slack_client=app.client,
    bot_user_id=bot_user_id,
    confluence_urls=confluence_urls
)
print(all([CONFLUENCE_URL, CONFLUENCE_TOKEN]))
if confluence_urls and all([CONFLUENCE_URL, CONFLUENCE_TOKEN]):
     logger.info("Attempting to initialize Confluence vector store on startup...")
     vector_store_manager.initialize_confluence_store() # Load from disk or build
else:
     logger.warning("Skipping Confluence store initialization due to missing config/URLs.")

# --- Initialize Tools ---
# Combine your common tools with the new Slack search tool
slack_search_tool = SlackChannelHistorySearchTool(vector_store_manager=vector_store_manager)
confluence_search_tool = ConfluenceSearchTool(vector_store_manager=vector_store_manager)
ALL_TOOLS = COMMON_TOOLS + [slack_search_tool, confluence_search_tool]
logger.info(f"Initialized tools: {[tool.name for tool in ALL_TOOLS]}")

# --- Agent Initialization ---
# Ensure the agent is initialized and available
try:
    agent_executor = init_agent(ALL_TOOLS)
    print("Langchain agent initialized successfully.")
except Exception as e:
    print(f"Error initializing Langchain agent: {e}")
    agent_executor = None # Set to None if initialization fails


@app.event("message")
def message_handler(message, logger):
    """Handles incoming messages to add them to the vector store."""
    channel_id = message.get('channel')
    user_id = message.get('user')
    message_ts = message.get('ts')
    event_type = message.get('type')
    subtype = message.get('subtype')
    text = message.get('text', '')

    if not text or subtype in ['bot_message', 'channel_join', 'channel_leave', 'message_deleted', 'message_changed']:
        return

    logger.info(f"Received user message in {channel_id} from {user_id} at {message_ts}. Processing for vector store.")

    try:
        message_doc = _create_document_from_slack_message(message, channel_id, app.client)
        vector_store_manager.add_slack_message(channel_id, message_doc)
    except Exception as e:
        logger.exception(f"Error processing message for vector store in {channel_id}: {e}")

# --- Event Handler for App Mentions ---
@app.event("app_mention")
def mention_handler(event, say, client, logger):
    """Handles mentions of the bot."""

    if not agent_executor:
        say("Sorry, my agent brain isn't working right now. Please try again later.")
        return

    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text', '')
    message_ts = event.get('ts') # Timestamp of the user's message
    thread_ts = event.get('thread_ts', None) # Timestamp of the thread (if exists)

    # --- 1. Add initial reaction ---
    try:
        client.reactions_add(
            channel=channel_id,
            name="eyes",
            timestamp=message_ts
        )
    except Exception as e:
        logger.error(f"Error adding reaction: {e}")

    try:
        # --- 2. Fetch and sort thread history (if applicable) ---
        chat_history = []
        if thread_ts: # If the message is part of a thread
            try:
                # Fetch all messages in the thread
                replies_response = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts # Use thread_ts to get the whole thread
                )
                messages = replies_response.get('messages', [])

                # Sort messages by timestamp (Slack usually returns them sorted)
                messages.sort(key=lambda x: float(x['ts']))

                # Process messages *before* the current one for history
                # The last message in 'messages' is the current mention event
                for message in messages[:-1]: # Exclude the last message (the current one)
                    msg_text = message.get('text', '')
                    msg_user = message.get('user')
                    if msg_user == bot_user_id:
                        chat_history.append(AIMessage(content=msg_text))
                    else:
                        # Include messages from other users/bots as HumanMessage
                        chat_history.append(HumanMessage(content=msg_text))
                logger.info(f"Reconstructed chat history with {len(chat_history)} messages.")

            except Exception as e:
                logger.error(f"Error fetching thread replies: {e}")
                # Proceed without history if fetching fails
                chat_history = []
        else:
            # Not in a thread, history is empty
            logger.info("Mention is not in a thread, starting with empty history.")
            pass # chat_history is already []

        # --- 3. Prepare input and invoke agent ---

        # Clean the input text to remove the bot mention (e.g., "<@U123ABC> ")
        if bot_user_id:
             mention_pattern = f"<@{bot_user_id}>"
             # Remove the mention and any leading/trailing whitespace
             input_text = text.replace(mention_pattern, "").strip()
        else:
             # Fallback if bot_user_id failed - might leave the mention text
             input_text = text.strip()

        if not input_text:
             # Handle cases where the message only contained the mention
             reply_text = "Yes? How can I help you?"
        else:
            logger.info(f"Invoking agent with input: '{input_text}' and history.")
            try:
                # Invoke the agent
                agent_response = agent_executor.invoke({
                    "input": input_text,
                    "chat_history": chat_history,
                    "channel_id": channel_id,
                })

                # --- 4. Parse output ---
                # Assuming the final answer is in the 'output' key
                reply_text = agent_response.get('output', "Sorry, I couldn't process that.")
                logger.info(f"Agent response: {reply_text}")
            except Exception as e:
                logger.error(f"Error invoking agent: {e}")
                reply_text = f"An error occurred while processing your request: {e}"
        
        # --- Post-process reply_text to add permalinks ---
        processed_reply_text = reply_text
        try:
            # Regex to find potential Slack timestamps (e.g., 1745507846.712679)
            # Word boundaries (\b) help avoid matching parts of other numbers
            timestamp_pattern = r'\b(\d{10}\.\d{6})\b'
            found_timestamps = re.findall(timestamp_pattern, reply_text)

            processed_timestamps = set() # Avoid processing the same timestamp multiple times
            if found_timestamps:
                logger.info(f"Found potential timestamps in LLM response: {found_timestamps}")
                for ts in found_timestamps:
                    if ts in processed_timestamps:
                        continue # Skip if already processed

                    logger.debug(f"Attempting to get permalink for ts: {ts} in channel: {channel_id}")
                    try:
                        # Call chat.getPermalink using the client from the handler context
                        permalink_response = client.chat_getPermalink(
                            channel=channel_id,
                            message_ts=ts
                        )
                        if permalink_response.get("ok"):
                            permalink = permalink_response.get("permalink")
                            # Replace the timestamp with Slack's link format <URL|Text>
                            link_text = ts # Or use more descriptive text like "[link]"
                            slack_link = f"<{permalink}|{link_text}>"
                            # Replace all occurrences of this specific timestamp
                            processed_reply_text = processed_reply_text.replace(ts, slack_link)
                            processed_timestamps.add(ts)
                            logger.info(f"Successfully replaced timestamp {ts} with link.")
                        else:
                            logger.warning(f"Failed to get permalink for ts {ts}: {permalink_response.get('error')}")
                            processed_timestamps.add(ts) # Mark as processed even if failed

                    except SlackApiError as permalink_error:
                        # Handle cases like message_not_found specifically if needed
                        if permalink_error.response.get("error") == "message_not_found":
                            logger.warning(f"Message with ts {ts} not found in channel {channel_id} for permalink.")
                        else:
                            logger.error(f"Slack API error getting permalink for ts {ts}: {permalink_error}")
                        processed_timestamps.add(ts) # Mark as processed even if failed
                    except Exception as general_error:
                        logger.error(f"Unexpected error getting permalink for ts {ts}: {general_error}")
                        processed_timestamps.add(ts) # Mark as processed even if failed

        except Exception as regex_error:
            logger.error(f"Error during timestamp processing regex: {regex_error}")
            # Fallback to using the original reply_text if regex fails
            processed_reply_text = reply_text
        # --- End Post-processing ---

        # --- 4. (cont.) Reply back in thread ---
        reply_thread_ts = thread_ts if thread_ts else message_ts # Reply in existing thread or start new one
        say(text=processed_reply_text, thread_ts=reply_thread_ts)

    except Exception as e:
        logger.error(f"Error in mention handler main block: {e}")
        try:
            # Try to send an error message back to the user
            reply_thread_ts = thread_ts if thread_ts else message_ts
            say(text=f"Sorry, an unexpected error occurred: {e}", thread_ts=reply_thread_ts)
        except Exception as say_e:
            logger.error(f"Failed to send error message to Slack: {say_e}")

    finally:
        # --- 5. Remove the eyes reaction ---
        try:
            client.reactions_remove(
                channel=channel_id,
                name="eyes",
                timestamp=message_ts
            )
        except Exception as e:
            logger.error(f"Error removing reaction: {e}")

# --- Graceful Shutdown Hook ---
def cleanup():
    logger.info("Running cleanup before exit...")
    vector_store_manager.save_all_stores() # Save stores on exit
    logger.info("Cleanup finished.")

atexit.register(cleanup)

# --- Main Execution ---
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()