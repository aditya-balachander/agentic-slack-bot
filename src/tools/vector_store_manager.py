# src/vector_store_manager.py
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from slack_sdk import WebClient

from src.tools.slack_loader import load_slack_channel_history

# --- Placeholder Constants ---
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=100
DEFAULT_RETRIEVAL_K=5
VECTOR_STORE_SAVE_DIR = Path("./vector_stores")

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages channel-specific FAISS vector stores and retrievers."""

    def __init__(
        self,
        embeddings: Embeddings,
        slack_client: WebClient,
        bot_user_id: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        save_dir: Path = VECTOR_STORE_SAVE_DIR
    ):
        self.embeddings = embeddings
        self.slack_client = slack_client
        self.bot_user_id = bot_user_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active vector stores and retrievers
        self._vector_stores: Dict[str, FAISS] = {}
        self._retrievers: Dict[str, BaseRetriever] = {}

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"VectorStoreManager initialized. Saving stores to: {self.save_dir.resolve()}")
        # Optional: Load existing stores on initialization
        self._load_all_stores_from_disk()


    def _get_store_path(self, channel_id: str) -> Path:
        """Gets the expected save path for a channel's vector store."""
        return self.save_dir / f"faiss_index_{channel_id}"

    def _load_store_from_disk(self, channel_id: str) -> Optional[FAISS]:
        """Loads a single FAISS index from disk if it exists."""
        store_path = self._get_store_path(channel_id)
        if store_path.exists():
            try:
                logger.info(f"Loading existing FAISS index for channel {channel_id} from {store_path}")
                # FAISS.load_local requires folder_path and embeddings
                vector_store = FAISS.load_local(str(store_path), self.embeddings, allow_dangerous_deserialization=True) # Set allow_dangerous_deserialization=True if loading pickle files
                logger.info(f"Successfully loaded index for {channel_id}.")
                return vector_store
            except Exception as e:
                logger.exception(f"Failed to load FAISS index for {channel_id} from {store_path}: {e}. Will rebuild if needed.")
                return None
        else:
            logger.debug(f"No existing index found at {store_path} for channel {channel_id}.")
            return None

    def _load_all_stores_from_disk(self):
        """Loads all detectable FAISS stores from the save directory."""
        logger.info(f"Scanning {self.save_dir} for existing vector stores...")
        loaded_count = 0
        for item in self.save_dir.iterdir():
            if item.is_dir() and item.name.startswith("faiss_index_"):
                channel_id = item.name.replace("faiss_index_", "")
                if channel_id not in self._vector_stores: # Avoid reloading if already in memory
                    store = self._load_store_from_disk(channel_id)
                    if store:
                        self._vector_stores[channel_id] = store
                        self._retrievers[channel_id] = store.as_retriever(search_kwargs={"k": self.retrieval_k})
                        loaded_count += 1
        logger.info(f"Finished loading existing stores. Loaded {loaded_count} stores.")


    def _save_store_to_disk(self, channel_id: str):
        """Saves the FAISS index for a channel to disk."""
        if channel_id in self._vector_stores:
            store_path = self._get_store_path(channel_id)
            try:
                logger.info(f"Saving FAISS index for channel {channel_id} to {store_path}")
                self._vector_stores[channel_id].save_local(str(store_path))
                logger.info(f"Successfully saved index for {channel_id}.")
            except Exception as e:
                logger.exception(f"Failed to save FAISS index for {channel_id} to {store_path}: {e}")
        else:
            logger.warning(f"Attempted to save store for channel {channel_id}, but it's not loaded.")

    def initialize_channel_store(self, channel_id: str, force_reload: bool = False):
        """
        Initializes the vector store for a channel by fetching history.
        Loads from disk if available unless force_reload is True.
        """
        if not force_reload and channel_id in self._retrievers:
            logger.debug(f"Retriever for channel {channel_id} already exists.")
            return # Already initialized

        # Try loading from disk first unless forced
        if not force_reload:
             existing_store = self._load_store_from_disk(channel_id)
             if existing_store:
                 self._vector_stores[channel_id] = existing_store
                 self._retrievers[channel_id] = existing_store.as_retriever(search_kwargs={"k": self.retrieval_k})
                 logger.info(f"Initialized retriever for {channel_id} from disk.")
                 return

        # If not loaded from disk or force_reload is True, fetch from Slack
        logger.info(f"Initializing vector store for channel {channel_id} by fetching history...")
        documents = load_slack_channel_history(channel_id, self.slack_client, self.bot_user_id)

        if not documents:
            logger.warning(f"No documents found or loaded for channel {channel_id}. Cannot create retriever.")
            # Optionally create an empty store or handle this case as needed
            # self._vector_stores[channel_id] = FAISS.from_texts(["placeholder"], self.embeddings) # Example empty store
            return

        logger.info(f"Splitting {len(documents)} documents for channel {channel_id}...")
        splits = self._text_splitter.split_documents(documents)
        logger.info(f"Generated {len(splits)} chunks for channel {channel_id}.")

        if not splits:
            logger.warning(f"No chunks generated after splitting for channel {channel_id}. Cannot create retriever.")
            return

        try:
            logger.info(f"Creating FAISS index for {len(splits)} chunks for channel {channel_id}...")
            # Use from_documents to create and populate in one step
            vector_store = FAISS.from_documents(splits, self.embeddings)
            self._vector_stores[channel_id] = vector_store
            self._retrievers[channel_id] = vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
            logger.info(f"Successfully created retriever for channel {channel_id}.")
            # Save the newly created store
            self._save_store_to_disk(channel_id)

        except Exception as e:
            logger.exception(f"Failed to create FAISS index for channel {channel_id}: {e}")


    def add_message(self, channel_id: str, message_doc: Document):
        """Adds a new message document to the appropriate channel's vector store."""
        if channel_id not in self._vector_stores:
            logger.warning(f"Vector store for channel {channel_id} not initialized. Initializing first...")
            self.initialize_channel_store(channel_id)
            # If initialization failed, the store might still not exist
            if channel_id not in self._vector_stores:
                 logger.error(f"Failed to initialize store for {channel_id}. Cannot add message.")
                 return

        vector_store = self._vector_stores[channel_id]

        # Split the single message document (optional, depends on message length vs chunk size)
        # If messages are short, splitting might not be necessary or even desired.
        # For simplicity here, we'll split even single messages.
        splits = self._text_splitter.split_documents([message_doc])

        if not splits:
            logger.warning(f"Message document for {channel_id} resulted in no splits. Not adding.")
            return

        try:
            # Add the new document chunks to the existing index
            # The IDs returned are internal to FAISS, may not be needed directly here
            vector_store.add_documents(splits)
            logger.info(f"Added {len(splits)} chunk(s) from new message in {channel_id} to vector store.")
            # Consider saving periodically or on shutdown, not after every message for performance
            # self._save_store_to_disk(channel_id) # Saving after every message can be slow

        except Exception as e:
            logger.exception(f"Failed to add message document to vector store for channel {channel_id}: {e}")

    def get_retriever(self, channel_id: str) -> Optional[BaseRetriever]:
        """Gets the retriever for the specified channel, initializing if necessary."""
        if channel_id not in self._retrievers:
            logger.info(f"Retriever for channel {channel_id} not found in cache. Attempting initialization.")
            self.initialize_channel_store(channel_id) # Attempt to load or build

        retriever = self._retrievers.get(channel_id)
        if not retriever:
             logger.warning(f"Retriever for {channel_id} could not be initialized or found.")
        return retriever

    def save_all_stores(self):
        """Saves all currently loaded vector stores to disk."""
        logger.info(f"Saving all {len(self._vector_stores)} loaded vector stores...")
        for channel_id in self._vector_stores.keys():
            self._save_store_to_disk(channel_id)
        logger.info("Finished saving all stores.")