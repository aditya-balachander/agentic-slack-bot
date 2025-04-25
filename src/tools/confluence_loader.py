import os
import logging
import requests 
from typing import List, Optional, Dict, Any
from langchain.docstore.document import Document
import urllib3 
import html2text 
from dotenv import load_dotenv 
import re 

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration from Environment Variables ---
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

def extract_page_id(link):
  """
  Extracts the page ID from a Confluence page link.

  Args:
    link: The Confluence page link (string).

  Returns:
    The page ID (string) if found, otherwise None.
  """
  match = re.search(r"/pages/(\d+)/", link)
  if match:
    return match.group(1)
  return None

def _fetch_page_content(page_id: str, session: requests.Session, base_url: str) -> Optional[Dict[str, Any]]:
    """Fetches content for a single page ID using the Confluence REST API."""
    # Construct the API endpoint URL
    # Expand body.storage for raw HTML content, version for info, space for context
    api_url = f"{base_url.rstrip('/')}/rest/api/content/{page_id}?expand=body.storage,version,space,history"
    logger.debug(f"Fetching Confluence page content from: {api_url}")
    try:
        response = session.get(api_url, timeout=20) # Increased timeout for potentially large pages
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        page_data = response.json()
        # Basic check for expected content
        if 'body' in page_data and 'storage' in page_data['body'] and 'value' in page_data['body']['storage']:
            return page_data
        else:
            logger.warning(f"Page {page_id} fetched successfully but missing expected content structure (body.storage.value).")
            return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching page {page_id}: {http_err}")
        # Specific handling for common errors
        if http_err.response.status_code == 404:
            logger.warning(f"Page {page_id} not found.")
        elif http_err.response.status_code == 403:
            logger.warning(f"Permission denied for page {page_id}. Check token permissions.")
        return None
    except requests.exceptions.RequestException as req_err:
        # Handles SSLError, ConnectionError, Timeout, etc.
        logger.error(f"Request error fetching page {page_id}: {req_err}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching page {page_id}: {e}")
        return None


def load_confluence_pages_from_urls(page_urls: List[str]) -> List[Document]:
    """
    Loads content from a list of Confluence page URLs using direct REST API calls
    with a configured requests session (handling SSL verification and authentication).

    Args:
        page_urls: A list of full Confluence page URLs to load.

    Returns:
        A list of Langchain Document objects, each representing content
        from a Confluence page, with metadata including the source URL.
    """
    if not all([CONFLUENCE_URL, CONFLUENCE_TOKEN]):
        logger.error("CONFLUENCE_URL or CONFLUENCE_API_TOKEN (containing the PAT) environment variables not set. Cannot load pages.")
        return []
    if not page_urls:
        logger.warning("No Confluence page URLs provided to load.")
        return []

    all_docs: List[Document] = []
    logger.info(f"Attempting to load {len(page_urls)} Confluence page(s) via REST API...")

    try:
        # --- Configure requests session for SSL handling and Authentication ---
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {CONFLUENCE_TOKEN}", "Accept": "application/json"})
        logger.info("Configured session with Bearer Token for Confluence authentication.")
        session.verify = False # WARNING: Insecure, bypasses SSL check
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.warning("SSL verification is disabled for Confluence requests. This is insecure.")

        # --- Diagnostic Step (Optional but recommended) ---
        test_url = CONFLUENCE_URL
        logger.info(f"Attempting direct request to {test_url} using configured session to test connectivity...")
        try:
            response = session.get(test_url, timeout=10)
            response.raise_for_status()
            logger.info(f"Direct session request to {test_url} SUCCEEDED (Status Code: {response.status_code}). Session configured correctly.")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Direct session request to {test_url} FAILED: {req_err}. Aborting Confluence load.")
            return [] # Stop if basic connectivity fails
        # --- End Diagnostic Step ---

        # --- Page ID Extraction Logic (remains the same, may need adjustment) ---
        page_ids_or_keys = []
        valid_urls_for_metadata = {}
        for url in page_urls:
             try:
                 page_id = extract_page_id(url)
                 if page_id.isdigit():
                      page_ids_or_keys.append(page_id)
                      valid_urls_for_metadata[page_id] = url
                 else:
                      logger.warning(f"Could not reliably extract Page ID from URL: {url}. Skipping for now.")
             except Exception:
                 logger.warning(f"Error processing URL for Page ID: {url}. Skipping.")

        if not page_ids_or_keys:
             logger.error("No valid Confluence Page IDs extracted or provided. Cannot load documents.")
             return []

        logger.info(f"Attempting to load Confluence content for Page IDs: {page_ids_or_keys} using REST API...")

        # --- Load Documents via REST API ---
        html_converter = html2text.HTML2Text()
        html_converter.ignore_links = False # Keep links in converted text
        html_converter.ignore_images = True # Ignore images

        for page_id in page_ids_or_keys:
            page_data = _fetch_page_content(page_id, session, CONFLUENCE_URL)
            if page_data:
                try:
                    title = page_data.get('title', f"Page {page_id}")
                    html_content = page_data.get('body', {}).get('storage', {}).get('value', '')
                    space_key = page_data.get('space', {}).get('key')
                    version_num = page_data.get('version', {}).get('number')
                    last_modified = page_data.get('history', {}).get('lastUpdated', {}).get('when') # Example of getting more metadata

                    # Convert HTML content to Markdown for potentially cleaner text
                    # Alternatively, use BeautifulSoup for more complex parsing
                    try:
                        text_content = html_converter.handle(html_content)
                    except Exception as conversion_err:
                        logger.warning(f"Failed to convert HTML to text for page {page_id}: {conversion_err}. Using raw HTML.")
                        text_content = html_content # Fallback to raw HTML

                    # Use the original URL if we mapped it, otherwise construct one
                    source_url = valid_urls_for_metadata.get(page_id, f"{CONFLUENCE_URL.rstrip('/')}/pages/viewpage.action?pageId={page_id}")

                    metadata = {
                        'source': source_url,
                        'id': page_id,
                        'title': title,
                        'space': space_key,
                        'version': version_num,
                        'last_modified': last_modified,
                        # Add any other relevant metadata extracted from page_data
                    }
                    doc = Document(page_content=text_content, metadata=metadata)
                    all_docs.append(doc)
                    logger.debug(f"Successfully processed page ID {page_id} ('{title}')")

                except Exception as processing_err:
                    logger.exception(f"Error processing data for page {page_id}: {processing_err}")

        logger.info(f"Successfully loaded and processed {len(all_docs)} document(s) from Confluence via REST API.")

    except ImportError as ie:
         logger.error(f"ImportError loading Confluence dependencies: {ie}. Make sure 'requests', 'html2text' are installed.")
    except Exception as e:
        logger.exception(f"Failed during Confluence page loading process: {e}")

    return all_docs


if __name__ == "__main__":
    # Example usage
    test_urls = [
        "https://confluence.internal.salesforce.com/spaces/IN/pages/630894927/How+to+decommission+Delete+Falcon+Services"
    ]
    documents = load_confluence_pages_from_urls(test_urls)
    for doc in documents:
        print(f"Loaded document with title: {doc.metadata.get('title')}")
        print(f"Content: {doc.page_content}")  # Print first 100 characters of content