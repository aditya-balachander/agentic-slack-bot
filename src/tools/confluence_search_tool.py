# src/tools/confluence_search_tool.py
import logging
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from src.tools.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class ConfluenceSearchInput(BaseModel):
    """Input schema for the ConfluenceSearchTool."""
    query: str = Field(description="The search query to run against the indexed Confluence documents.")

class ConfluenceSearchTool(BaseTool):
    """Tool to search indexed Confluence documents for relevant information."""
    name: str = "confluence_document_search"
    description: str = (
        "Searches the indexed Confluence documentation for relevant information, procedures, or explanations. "
        "Use this when the user asks a question that is likely answered in the official documentation or knowledge base. "
        "Requires only the 'query' (what to search for)."
    )
    args_schema: Type[BaseModel] = ConfluenceSearchInput
    vector_store_manager: VectorStoreManager

    def _run(self, query: str) -> str:
        """Execute the search query against the Confluence document index."""
        logger.info(f"[TOOL RUN] Executing Confluence search for query: '{query}'")

        retriever = self.vector_store_manager.get_confluence_retriever()
        if not retriever:
            logger.error("Confluence retriever is not available.")
            # Attempt to initialize it - might be needed if accessed before startup init completes
            self.vector_store_manager.initialize_confluence_store()
            retriever = self.vector_store_manager.get_confluence_retriever()
            if not retriever:
                 return "Error: Confluence documentation index is not available or failed to initialize."


        try:
            results: List[Document] = retriever.get_relevant_documents(query)

            if not results:
                return f"No relevant information found in the indexed Confluence documents for query: '{query}'"

            # --- Format results clearly for the LLM, including source links ---
            formatted_results = []
            for doc in results:
                metadata = doc.metadata
                # Get source URL and title if available
                source_url = metadata.get('source', '#') # Default to '#' if no URL
                page_title = metadata.get('title', 'Confluence Document') # Get title if loader provided it

                # Create a Slack-compatible link if URL is valid
                link_display = f"<{source_url}|{page_title}>" if source_url != '#' else page_title

                formatted_results.append(
                    f"Source Document: {link_display}\n"
                    f"Relevant Content: ...{doc.page_content}..." # Keep content concise for LLM
                )

            # Limit the number of results shown?
            max_results_to_show = 3
            output_string = f"Found relevant information in Confluence documentation:\n\n" + "\n---\n".join(formatted_results[:max_results_to_show])
            if len(results) > max_results_to_show:
                output_string += f"\n\n...(found {len(results)} relevant sections in total)"

            return output_string
            # --- End Formatting ---

        except Exception as e:
            logger.exception(f"Error during Confluence similarity search: {e}")
            return f"Error performing search in Confluence documents: {e}"

    async def _arun(self, query: str) -> str:
        # Apply similar logic for async execution
        logger.info(f"[TOOL ARUN] Executing Confluence search for query: '{query}'")
        retriever = self.vector_store_manager.get_confluence_retriever()
        if not retriever:
            self.vector_store_manager.initialize_confluence_store()
            retriever = self.vector_store_manager.get_confluence_retriever()
            if not retriever:
                return "Error: Confluence documentation index is not available or failed to initialize (async)."

        try:
            results: List[Document] = await retriever.aget_relevant_documents(query)
            if not results:
                return f"No relevant information found in the indexed Confluence documents for query: '{query}' (async)"

            # --- Format results (Async) ---
            formatted_results = []
            for doc in results:
                metadata = doc.metadata
                source_url = metadata.get('source', '#')
                page_title = metadata.get('title', 'Confluence Document')
                link_display = f"<{source_url}|{page_title}>" if source_url != '#' else page_title
                formatted_results.append(
                    f"Source Document: {link_display}\n"
                    f"Relevant Content: ...{doc.page_content}..."
                )

            max_results_to_show = 3
            output_string = f"Found relevant information in Confluence documentation (async):\n\n" + "\n---\n".join(formatted_results[:max_results_to_show])
            if len(results) > max_results_to_show:
                output_string += f"\n\n...(found {len(results)} relevant sections in total)"
            return output_string
            # --- End Formatting ---

        except Exception as e:
            logger.exception(f"Error during async Confluence similarity search: {e}")
            return f"Error performing async search in Confluence documents: {e}"

