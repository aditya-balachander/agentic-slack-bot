import requests
from typing import Any, Dict, Type, Optional, List
import json
import os
from typing_extensions import List
import faiss
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL = "llmgateway__OpenAITextEmbedding3_Large"
DEFAULT_DOCS_FOLDER = "./src/docs"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 3

LOADER_MAPPING: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredFileLoader,
    ".html": UnstructuredFileLoader,
    ".htm": UnstructuredFileLoader,
    ".doc": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader,
}

class EinsteinEmbeddings(Embeddings):

    gateway_server: str = os.environ.get("EINSTEIN_GATEWAY_SERVER")
    gateway_path: str = "v1.0"
    api_key: str = os.environ.get("EINSTEIN_API_KEY")
    feature_id: str = "EinsteinDocsAnswers"
    app_context: str = "EinsteinGPT"
    core_tenant_id: str = "core/prod1/00DDu0000008cuqMAA"

    def __init__(
        self,
        model: str
    ):
        self.model = model
        self._client = requests.Session()
        self._api_url = f"https://{self.gateway_server}/{self.gateway_path}/embeddings"
        self._headers = self._build_headers()

    def _build_headers(self) -> Dict[str, str]:
        """Constructs the request headers."""
        headers = {
            "Authorization": f"API_KEY {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.feature_id:
            headers["x-client-feature-id"] = self.feature_id
        if self.app_context:
            headers["x-sfdc-app-context"] = self.app_context
        if self.core_tenant_id:
            headers["x-sfdc-core-tenant-id"] = self.core_tenant_id
        return headers

    def _call_api(self, texts: List[str]) -> Any:
        """Makes the actual POST request to the embeddings endpoint."""
        payload = {
            "input": texts,
            "model": self.model
        }

        try:
            response = self._client.post(
                self._api_url,
                headers=self._headers,
                json=payload,
                verify=False
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Einstein LLM Gateway: {e}")
            # Handle specific errors (e.g., connection, timeout) if needed
            if hasattr(e, 'response') and e.response is not None:
                 print(f"Response status code: {e.response.status_code}")
                 print(f"Response body: {e.response.text}")
            raise # Re-raise the exception after logging

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return [] # Handle empty input list

        num_texts = len(texts)
        response_data = self._call_api(texts)

        try:
            embedding_data_list = response_data.get("embeddings")
            if embedding_data_list is None:
                raise KeyError("Expected key 'embeddings' not found in API response.")
            if not isinstance(embedding_data_list, list):
                 raise ValueError("Value for key 'embeddings' is not a list.")

            # Initialize result list with None placeholders to place embeddings by index
            result_embeddings: List[Optional[List[float]]] = [None] * num_texts

            for item in embedding_data_list:
                if not isinstance(item, dict):
                    raise ValueError("Item within 'embeddings' list is not a dictionary.")

                embedding_vector = item.get("embedding")
                index = item.get("index")

                # Validate extracted data
                if embedding_vector is None:
                    raise KeyError(f"Missing 'embedding' key in item: {item}")
                if index is None:
                    raise KeyError(f"Missing 'index' key in item: {item}")
                if not isinstance(embedding_vector, list):
                     raise ValueError(f"Value for 'embedding' key is not a list in item: {item}")
                if not isinstance(index, int):
                    raise ValueError(f"Value for 'index' key is not an integer in item: {item}")
                if not (0 <= index < num_texts):
                     raise IndexError(f"Index {index} from response is out of bounds for input texts (count: {num_texts}).")
                if result_embeddings[index] is not None:
                     raise ValueError(f"Duplicate index {index} received in API response.")

                # Store the embedding at the correct index
                result_embeddings[index] = embedding_vector

            # Check if all embeddings were received and placed
            if None in result_embeddings:
                missing_indices = [i for i, emb in enumerate(result_embeddings) if emb is None]
                raise ValueError(f"API response did not contain embeddings for all input texts. Missing indices: {missing_indices}")

            # Cast is safe here because we checked for None above
            return [emb for emb in result_embeddings if emb is not None] # Or simply return result_embeddings after python 3.7+ type hinting

        except (KeyError, ValueError, IndexError) as e:
            print(f"Error processing API response structure: {e}")
            print(f"Received response data: {json.dumps(response_data, indent=2)}") # Pretty print for readability
            raise
        except Exception as e:
             print(f"An unexpected error occurred while processing the response: {e}")
             print(f"Received response data: {json.dumps(response_data, indent=2)}")
             raise


    def embed_query(self, text: str) -> List[float]:
        if not text:
            # Or return a default zero vector, depending on desired behavior
            raise ValueError("Cannot embed an empty query text.")

        embeddings_list = self.embed_documents([text])

        if len(embeddings_list) != 1:
            raise ValueError(f"Expected 1 embedding for the query, but received {len(embeddings_list)}.")

        return embeddings_list[0]

def load_documents_from_folder(folder_path: str = DEFAULT_DOCS_FOLDER) -> List[Document]:
    docs_path = Path(folder_path)
    if not docs_path.is_dir():
        logger.error(f"Document folder not found or is not a directory: {folder_path}")
        return []

    all_docs: List[Document] = []
    logger.info(f"Scanning document folder: {docs_path.resolve()}")

    loaded_files = 0
    skipped_files = 0
    for item_path in docs_path.iterdir():
        if item_path.is_file():
            file_suffix = item_path.suffix.lower()
            loader_class = LOADER_MAPPING.get(file_suffix)

            if loader_class:
                logger.debug(f"Attempting to load: {item_path.name} using {loader_class.__name__}")
                try:
                    # Pass file path as string to the loader instance
                    loader = loader_class(str(item_path))
                    loaded_file_docs = loader.load() # Load documents from the file

                    # Ensure the loader returned a list of Document objects
                    if isinstance(loaded_file_docs, list) and all(isinstance(d, Document) for d in loaded_file_docs):
                         all_docs.extend(loaded_file_docs)
                         logger.info(f"Successfully loaded {len(loaded_file_docs)} document(s) from {item_path.name}")
                         loaded_files += 1
                    else:
                         logger.warning(f"Loader for {item_path.name} did not return a valid list of Documents. Skipping.")
                         skipped_files += 1

                except Exception as e:
                    logger.error(f"Failed to load file {item_path.name}: {e}", exc_info=False) # Set exc_info=True for traceback
                    skipped_files += 1
            else:
                logger.debug(f"Skipping file with unsupported extension: {item_path.name}")
                skipped_files += 1

    logger.info(f"Document loading complete. Loaded: {loaded_files} files, Skipped/Failed: {skipped_files} files. Total documents: {len(all_docs)}")
    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    if not docs:
        logger.warning("No documents provided to split.")
        return []

    logger.info(f"Splitting {len(docs)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Splitting complete. Generated {len(splits)} chunks.")
    return splits


def create_retriever(
    documents: List[Document],
    embeddings: Embeddings,
    k: int = DEFAULT_RETRIEVAL_K
) -> Optional[VectorStoreRetriever]:

    logger.info(f"Creating FAISS index for {len(documents)} document chunks...")
    try:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        _ = vector_store.add_documents(documents=documents)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        logger.exception("Failed to create FAISS vector store") # Log full traceback
        return None

def setup_retriever_from_docs_folder(
     docs_folder: str = DEFAULT_DOCS_FOLDER,
     chunk_size: int = DEFAULT_CHUNK_SIZE,
     chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
     retrieval_k: int = DEFAULT_RETRIEVAL_K,
     embedding_instance: Optional[Embeddings] = None,
     embedding_model_name: str = MODEL
) -> Optional[VectorStoreRetriever]:
    
    if embedding_instance is None:
        embedding_instance = EinsteinEmbeddings(model=embedding_model_name)

    documents = load_documents_from_folder(docs_folder)
    all_splits = split_documents(documents, chunk_size, chunk_overlap)
    retriever = create_retriever(all_splits, embedding_instance, retrieval_k)
    return retriever