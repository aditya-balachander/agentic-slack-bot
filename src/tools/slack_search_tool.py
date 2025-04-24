# src/tools/slack_search_tool.py
import logging
from typing import Type, List
from pydantic import BaseModel, Field # For defining input schema
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from src.tools.vector_store_manager import VectorStoreManager # Correct import path

logger = logging.getLogger(__name__)

class SlackSearchInput(BaseModel):
    """Input schema for the SlackChannelHistorySearchTool."""
    query: str = Field(description="The search query to run against the channel history.")
    channel_id: str = Field(description="The ID of the Slack channel to search within.")

class SlackChannelHistorySearchTool(BaseTool):
    """Tool to search the message history of a specific Slack channel."""
    name: str = "slack_channel_history_search"
    description: str = (
        "Searches the message history of a specific Slack channel. "
        "Use this to find past discussions, mentions, or information within the channel. "
        "Requires 'query' (what to search for) and 'channel_id' (which channel to search)."
    )
    args_schema: Type[BaseModel] = SlackSearchInput
    vector_store_manager: VectorStoreManager # Pass the manager instance during initialization

    def _run(self, query: str, channel_id: str) -> str:
        """Execute the search query against the specified channel's history."""
        logger.info(f"Executing Slack history search in channel '{channel_id}' for query: '{query}'")
        if not channel_id:
             return "Error: Channel ID was not provided. Cannot search history."

        retriever = self.vector_store_manager.get_retriever(channel_id)
        if not retriever:
            # Attempt initialization if retriever wasn't ready (e.g., first message in channel)
            self.vector_store_manager.initialize_channel_store(channel_id)
            retriever = self.vector_store_manager.get_retriever(channel_id)

        if not retriever:
            return f"Error: Could not access or initialize message history for channel {channel_id}."

        try:
            results: List[Document] = retriever.get_relevant_documents(query)
            if not results:
                return f"No relevant messages found in channel {channel_id} for query: '{query}'"

            # --- Format results clearly for the LLM ---
            formatted_results = []
            for doc in results:
                metadata = doc.metadata
                # Use fetched user_name, fallback gracefully
                user_display = metadata.get('user_name', metadata.get('user_id', 'Unknown'))
                if metadata.get('user_id') and user_display != metadata.get('user_id'):
                    user_display = f"{user_display} ({metadata.get('user_id')})" # Add ID if name is different

                timestamp = metadata.get('message_ts', 'Unknown Timestamp')
                # Extract raw text content if possible (assuming specific format)
                raw_text = doc.page_content
                if f"User {user_display} at {timestamp}: " in raw_text:
                     raw_text = raw_text.split(f"User {user_display} at {timestamp}: ", 1)[1]

                formatted_results.append(
                    f"Message from: {user_display}\n"
                    f"Timestamp: {timestamp}\n" # Provide the exact timestamp
                    f"Content: {raw_text}"
                )

            return f"Found relevant messages in channel {channel_id}:\n\n" + "\n---\n".join(formatted_results)

        except Exception as e:
            logger.exception(f"Error during similarity search in channel {channel_id}: {e}")
            return f"Error performing search in channel {channel_id}: {e}"

    async def _arun(self, query: str, channel_id: str) -> str:
        # Basic async wrapper for now. Proper async requires async retriever methods.
        # Langchain's BaseRetriever has `aget_relevant_documents`
        logger.info(f"Executing Slack history search (async) in channel '{channel_id}' for query: '{query}'")
        if not channel_id:
             return "Error: Channel ID was not provided. Cannot search history."

        retriever = self.vector_store_manager.get_retriever(channel_id)
        if not retriever:
            # Async initialization might be complex, doing sync here for simplicity
            self.vector_store_manager.initialize_channel_store(channel_id)
            retriever = self.vector_store_manager.get_retriever(channel_id)

        if not retriever:
            return f"Error: Could not access or initialize message history for channel {channel_id} (async)."
        try:
            results: List[Document] = await retriever.aget_relevant_documents(query)
            if not results:
                 return f"No relevant messages found in channel {channel_id} for query: '{query}' (async)"

            # --- Format results clearly for the LLM (Async) ---
            formatted_results = []
            for doc in results:
                metadata = doc.metadata
                user_display = metadata.get('user_name', metadata.get('user_id', 'Unknown'))
                if metadata.get('user_id') and user_display != metadata.get('user_id'):
                    user_display = f"{user_display} ({metadata.get('user_id')})"
                timestamp = metadata.get('message_ts', 'Unknown Timestamp')
                raw_text = doc.page_content
                if f"User {user_display} at {timestamp}: " in raw_text:
                     raw_text = raw_text.split(f"User {user_display} at {timestamp}: ", 1)[1]

                formatted_results.append(
                    f"Message from: {user_display}\n"
                    f"Timestamp: {timestamp}\n"
                    f"Content: {raw_text}"
                )
            return f"Found relevant messages in channel {channel_id} (async):\n\n" + "\n---\n".join(formatted_results)
            # --- End Formatting ---

        except Exception as e:
            logger.exception(f"Error during async similarity search in channel {channel_id}: {e}")
            return f"Error performing async search in channel {channel_id}: {e}"
