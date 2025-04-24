import time
import logging
from typing import List, Dict, Any
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure logging as needed

# --- Constants ---
DEFAULT_PAGE_LIMIT = 200
MAX_MESSAGES_PER_CHANNEL = 10000
MESSAGE_FETCH_DELAY_SECONDS = 1.2 # Respect Slack Tier 3 rate limits (avg ~50/min)
USER_INFO_CACHE = {}

def _get_user_name(user_id: str, client: WebClient) -> str:
    """Fetches user name from Slack API, uses cache."""
    if not user_id or user_id == 'UNKNOWN_USER':
        return 'Unknown User'
    # Check cache first
    if user_id in USER_INFO_CACHE:
        return USER_INFO_CACHE[user_id]
    try:
        # Limit API calls by checking if it looks like a user ID
        if not user_id.startswith('U') and not user_id.startswith('W'):
             logger.debug(f"Skipping users.info call for likely non-user ID: {user_id}")
             return user_id # Return the ID itself if it doesn't look like a user

        logger.debug(f"Calling users.info for {user_id}")
        user_info_response = client.users_info(user=user_id)
        user_data = user_info_response.get('user', {})
        # Prefer real_name, fallback to name (display name), then ID
        name = user_data.get('real_name', user_data.get('name', user_id))
        # Cache the result
        USER_INFO_CACHE[user_id] = name
        return name
    except SlackApiError as e:
        # Handle specific errors like user_not_found if needed
        if e.response.get("error") == "user_not_found":
             logger.warning(f"User ID {user_id} not found.")
             USER_INFO_CACHE[user_id] = f"Unknown User ({user_id})" # Cache not found state
             return f"Unknown User ({user_id})"
        logger.error(f"Failed to get user info for {user_id} due to Slack API error: {e}. Using user ID.")
        return user_id # Fallback to ID on other errors
    except Exception as e:
        logger.error(f"Unexpected error fetching user info for {user_id}: {e}. Using user ID.")
        return user_id # Fallback to ID on other errors

def _create_document_from_slack_message(message: Dict[str, Any], channel_id: str, client: WebClient) -> Document:
    """Converts a Slack message dictionary into a Langchain Document."""
    text = message.get('text', '')
    user = message.get('user', 'UNKNOWN_USER')
    ts = message.get('ts', '') # Timestamp acts as a unique ID within the channel
    user_name = _get_user_name(user, client)
    # Add other potentially useful metadata
    metadata = {
        'source': f'slack_channel_{channel_id}',
        'channel_id': channel_id,
        'user_id': user,
        'user_name': user_name,
        'timestamp': ts,
        'message_ts': ts,
        'thread_ts': message.get('thread_ts', None)
    }
    page_content = f"User {user_name} at {ts}: {text}"
    return Document(page_content=page_content, metadata=metadata)

def load_slack_channel_history(
    channel_id: str,
    client: WebClient,
    bot_user_id: str,
    max_messages: int = MAX_MESSAGES_PER_CHANNEL,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> List[Document]:
    """
    Fetches message history for a Slack channel and converts to Langchain Documents.

    Args:
        channel_id: The ID of the channel to fetch history from.
        client: An initialized Slack WebClient.
        bot_user_id: The User ID of the bot to potentially filter its own messages if needed.
        max_messages: The maximum number of messages to fetch.
        page_limit: Number of messages per API call page.

    Returns:
        A list of Langchain Document objects representing the messages.
    """
    all_documents: List[Document] = []
    message_count = 0
    next_cursor = None
    logger.info(f"Starting history fetch for channel {channel_id}...")

    while message_count < max_messages:
        try:
            response = client.conversations_history(
                channel=channel_id,
                limit=page_limit,
                cursor=next_cursor
            )

            if not response.get("ok"):
                logger.error(f"Slack API error fetching history for {channel_id}: {response.get('error', 'Unknown error')}")
                break

            messages = response.get('messages', [])
            if not messages:
                logger.info(f"No more messages found for channel {channel_id}.")
                break # No more messages

            for message in messages:
                # Basic filtering: Skip message subtypes often not useful for retrieval
                # like channel joins, leaves, topic changes etc. Add more as needed.
                subtype = message.get('subtype')
                if subtype in ['channel_join', 'channel_leave', 'channel_topic', 'channel_purpose']:
                     continue
                # Optionally filter out bot's own messages if desired
                # if message.get('user') == bot_user_id:
                #    continue

                if message.get('text'): # Only process messages with text content
                    doc = _create_document_from_slack_message(message, channel_id, client)
                    all_documents.append(doc)
                    message_count += 1
                    if message_count >= max_messages:
                        break

            logger.info(f"Fetched page for {channel_id}. Total messages so far: {message_count}")

            # Check for pagination
            next_cursor = response.get('response_metadata', {}).get('next_cursor')
            if not next_cursor:
                logger.info(f"No next cursor, history fetch complete for {channel_id}.")
                break # End of history

            # IMPORTANT: Respect Slack rate limits
            time.sleep(MESSAGE_FETCH_DELAY_SECONDS)

        except SlackApiError as e:
            logger.error(f"Slack API Error during history fetch for {channel_id}: {e.response['error']}")
            if e.response.get("retry_after"):
                retry_delay = int(e.response["retry_after"])
                logger.warning(f"Rate limit hit. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                time.sleep(5) # Default backoff
        except Exception as e:
            logger.exception(f"Unexpected error fetching history for {channel_id}: {e}")
            break # Stop fetching on unexpected errors

    logger.info(f"Finished history fetch for channel {channel_id}. Total documents loaded: {len(all_documents)}")
    # Slack returns newest first, reverse to get chronological order for splitting/context
    all_documents.reverse()
    return all_documents