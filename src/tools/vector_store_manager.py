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

# --- Project Imports ---
from src.tools.slack_loader import load_slack_channel_history, _create_document_from_slack_message
from src.tools.confluence_loader import load_confluence_pages_from_urls

# --- Placeholder Constants ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150 # Increased overlap might help with context
DEFAULT_RETRIEVAL_K = 5
VECTOR_STORE_SAVE_DIR = Path("./vector_stores") # Directory to save/load stores

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages channel-specific Slack FAISS vector stores and a global Confluence store."""

    def __init__(
        self,
        embeddings: Embeddings,
        slack_client: "WebClient", # Type hint requires forward reference or import
        bot_user_id: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        save_dir: Path = VECTOR_STORE_SAVE_DIR,
        confluence_urls: Optional[List[str]] = None, # Add confluence URLs
    ):
        self.embeddings = embeddings
        self.slack_client = slack_client
        self.bot_user_id = bot_user_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.confluence_urls = confluence_urls if confluence_urls else []

        # --- Slack Stores ---
        self._slack_vector_stores: Dict[str, FAISS] = {}
        self._slack_retrievers: Dict[str, BaseRetriever] = {}

        # --- Confluence Store ---
        self._confluence_vector_store: Optional[FAISS] = None
        self._confluence_retriever: Optional[BaseRetriever] = None
        self._confluence_store_path = self.save_dir / "faiss_index_confluence" # Specific path

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""], # Standard separators
            length_function=len
        )
        logger.info(f"VectorStoreManager initialized. Saving stores to: {self.save_dir.resolve()}")

        # --- Load existing stores on initialization ---
        self._load_all_slack_stores_from_disk()
        self._load_confluence_store_from_disk() # Load confluence store


    # === Slack Store Methods ===

    def _get_slack_store_path(self, channel_id: str) -> Path:
        """Gets the expected save path for a Slack channel's vector store."""
        return self.save_dir / f"faiss_index_slack_{channel_id}"

    def _load_slack_store_from_disk(self, channel_id: str) -> Optional[FAISS]:
        """Loads a single Slack FAISS index from disk if it exists."""
        store_path = self._get_slack_store_path(channel_id)
        if store_path.exists():
            try:
                logger.info(f"Loading existing Slack FAISS index for channel {channel_id} from {store_path}")
                vector_store = FAISS.load_local(str(store_path), self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Successfully loaded Slack index for {channel_id}.")
                return vector_store
            except Exception as e:
                logger.exception(f"Failed to load Slack FAISS index for {channel_id} from {store_path}: {e}. Will rebuild if needed.")
                return None
        else:
            # logger.debug(f"No existing Slack index found at {store_path} for channel {channel_id}.")
            return None

    def _load_all_slack_stores_from_disk(self):
        """Loads all detectable Slack FAISS stores from the save directory."""
        logger.info(f"Scanning {self.save_dir} for existing Slack vector stores...")
        loaded_count = 0
        for item in self.save_dir.iterdir():
            if item.is_dir() and item.name.startswith("faiss_index_slack_"):
                channel_id = item.name.replace("faiss_index_slack_", "")
                if channel_id not in self._slack_vector_stores:
                    store = self._load_slack_store_from_disk(channel_id)
                    if store:
                        self._slack_vector_stores[channel_id] = store
                        self._slack_retrievers[channel_id] = store.as_retriever(search_kwargs={"k": self.retrieval_k})
                        loaded_count += 1
        logger.info(f"Finished loading existing Slack stores. Loaded {loaded_count} stores.")

    def _save_slack_store_to_disk(self, channel_id: str):
        """Saves the Slack FAISS index for a channel to disk."""
        if channel_id in self._slack_vector_stores:
            store_path = self._get_slack_store_path(channel_id)
            try:
                logger.info(f"Saving Slack FAISS index for channel {channel_id} to {store_path}")
                self._slack_vector_stores[channel_id].save_local(str(store_path))
                logger.info(f"Successfully saved Slack index for {channel_id}.")
            except Exception as e:
                logger.exception(f"Failed to save Slack FAISS index for {channel_id} to {store_path}: {e}")
        else:
            logger.warning(f"Attempted to save Slack store for channel {channel_id}, but it's not loaded.")

    def initialize_slack_channel_store(self, channel_id: str, force_reload: bool = False):
        """Initializes the Slack vector store for a channel by fetching history."""
        if not force_reload and channel_id in self._slack_retrievers:
            logger.debug(f"Slack retriever for channel {channel_id} already exists.")
            return

        if not force_reload:
             existing_store = self._load_slack_store_from_disk(channel_id)
             if existing_store:
                 self._slack_vector_stores[channel_id] = existing_store
                 self._slack_retrievers[channel_id] = existing_store.as_retriever(search_kwargs={"k": self.retrieval_k})
                 logger.info(f"Initialized Slack retriever for {channel_id} from disk.")
                 return

        logger.info(f"Initializing Slack vector store for channel {channel_id} by fetching history...")
        # Pass the slack_client here
        documents = load_slack_channel_history(channel_id, self.slack_client, self.bot_user_id)

        if not documents:
            logger.warning(f"No Slack documents found or loaded for channel {channel_id}. Cannot create retriever.")
            return

        logger.info(f"Splitting {len(documents)} Slack documents for channel {channel_id}...")
        splits = self._text_splitter.split_documents(documents)
        logger.info(f"Generated {len(splits)} Slack chunks for channel {channel_id}.")

        if not splits:
            logger.warning(f"No Slack chunks generated after splitting for channel {channel_id}.")
            return

        try:
            logger.info(f"Creating Slack FAISS index for {len(splits)} chunks for channel {channel_id}...")
            vector_store = FAISS.from_documents(splits, self.embeddings)
            self._slack_vector_stores[channel_id] = vector_store
            self._slack_retrievers[channel_id] = vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
            logger.info(f"Successfully created Slack retriever for channel {channel_id}.")
            self._save_slack_store_to_disk(channel_id)
        except Exception as e:
            logger.exception(f"Failed to create Slack FAISS index for channel {channel_id}: {e}")

    def add_slack_message(self, channel_id: str, message_dict: Dict):
        """Adds a new Slack message to the appropriate channel's vector store."""
        if channel_id not in self._slack_vector_stores:
            logger.warning(f"Slack vector store for channel {channel_id} not initialized. Initializing first...")
            self.initialize_slack_channel_store(channel_id)
            if channel_id not in self._slack_vector_stores:
                 logger.error(f"Failed to initialize Slack store for {channel_id}. Cannot add message.")
                 return

        vector_store = self._slack_vector_stores[channel_id]
        # Create document, passing the client
        message_doc = _create_document_from_slack_message(message_dict, channel_id, self.slack_client)
        splits = self._text_splitter.split_documents([message_doc])

        if not splits:
            logger.warning(f"Slack message document for {channel_id} resulted in no splits. Not adding.")
            return

        try:
            vector_store.add_documents(splits)
            logger.info(f"Added {len(splits)} chunk(s) from new Slack message in {channel_id} to vector store.")
            # Consider periodic saving instead of immediate saving
            # self._save_slack_store_to_disk(channel_id)
        except Exception as e:
            logger.exception(f"Failed to add Slack message document to vector store for channel {channel_id}: {e}")

    def get_slack_retriever(self, channel_id: str) -> Optional[BaseRetriever]:
        """Gets the Slack retriever for the specified channel, initializing if necessary."""
        if channel_id not in self._slack_retrievers:
            logger.info(f"Slack retriever for channel {channel_id} not found in cache. Attempting initialization.")
            self.initialize_slack_channel_store(channel_id)

        retriever = self._slack_retrievers.get(channel_id)
        if not retriever:
             logger.warning(f"Slack retriever for {channel_id} could not be initialized or found.")
        return retriever

    # === Confluence Store Methods ===

    def _load_confluence_store_from_disk(self) -> bool:
        """Loads the Confluence FAISS index from disk if it exists."""
        if self._confluence_store_path.exists():
            try:
                logger.info(f"Loading existing Confluence FAISS index from {self._confluence_store_path}")
                vector_store = FAISS.load_local(str(self._confluence_store_path), self.embeddings, allow_dangerous_deserialization=True)
                self._confluence_vector_store = vector_store
                self._confluence_retriever = vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
                logger.info("Successfully loaded Confluence index from disk.")
                return True
            except Exception as e:
                logger.exception(f"Failed to load Confluence FAISS index from {self._confluence_store_path}: {e}. Will rebuild if needed.")
                self._confluence_vector_store = None
                self._confluence_retriever = None
                return False
        else:
            logger.info(f"No existing Confluence index found at {self._confluence_store_path}.")
            return False

    def _save_confluence_store_to_disk(self):
        """Saves the Confluence FAISS index to disk."""
        if self._confluence_vector_store:
            try:
                logger.info(f"Saving Confluence FAISS index to {self._confluence_store_path}")
                self._confluence_vector_store.save_local(str(self._confluence_store_path))
                logger.info("Successfully saved Confluence index.")
            except Exception as e:
                logger.exception(f"Failed to save Confluence FAISS index to {self._confluence_store_path}: {e}")
        else:
            logger.warning("Attempted to save Confluence store, but it's not loaded.")

    def initialize_confluence_store(self, force_reload: bool = False):
        """
        Initializes the Confluence vector store by loading pages specified
        in the constructor or environment. Loads from disk if available unless force_reload.
        """
        if not force_reload and self._confluence_retriever:
            logger.info("Confluence retriever already initialized.")
            return

        # Try loading from disk first unless forced
        if not force_reload:
             if self._load_confluence_store_from_disk():
                 logger.info("Initialized Confluence retriever from disk.")
                 return
             else:
                  logger.info("Confluence index not found on disk or failed to load. Will build from source.")


        # If not loaded from disk or force_reload is True, fetch from Confluence
        if not self.confluence_urls:
             logger.warning("No Confluence URLs configured. Cannot initialize Confluence store.")
             return

        logger.info(f"Initializing Confluence vector store from URLs: {self.confluence_urls}")
        documents = load_confluence_pages_from_urls(self.confluence_urls)

        if not documents:
            logger.error("No documents loaded from Confluence. Cannot create retriever.")
            return

        logger.info(f"Splitting {len(documents)} Confluence documents...")
        splits = self._text_splitter.split_documents(documents)
        logger.info(f"Generated {len(splits)} Confluence chunks.")

        if not splits:
            logger.warning("No Confluence chunks generated after splitting. Cannot create retriever.")
            return

        try:
            logger.info(f"Creating Confluence FAISS index for {len(splits)} chunks...")
            vector_store = FAISS.from_documents(splits, self.embeddings)
            self._confluence_vector_store = vector_store
            self._confluence_retriever = vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
            logger.info("Successfully created Confluence retriever.")
            # Save the newly created store
            self._save_confluence_store_to_disk()

        except Exception as e:
            logger.exception(f"Failed to create Confluence FAISS index: {e}")
            self._confluence_vector_store = None
            self._confluence_retriever = None


    def get_confluence_retriever(self) -> Optional[BaseRetriever]:
        """Gets the Confluence retriever, initializing if necessary."""
        if not self._confluence_retriever:
            logger.warning("Confluence retriever not initialized. Attempting initialization.")
            # Attempt to initialize (will try loading from disk first)
            self.initialize_confluence_store()

        if not self._confluence_retriever:
             logger.error("Confluence retriever could not be initialized or found.")

        return self._confluence_retriever


    # === General Methods ===

    def save_all_stores(self):
        """Saves all currently loaded vector stores (Slack and Confluence) to disk."""
        logger.info("Saving all vector stores...")
        # Save Slack stores
        logger.info(f"Saving {len(self._slack_vector_stores)} loaded Slack stores...")
        for channel_id in self._slack_vector_stores.keys():
            self._save_slack_store_to_disk(channel_id)
        # Save Confluence store
        self._save_confluence_store_to_disk()
        logger.info("Finished saving all stores.")

