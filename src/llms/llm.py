import requests
import os
import logging
import time
import json
from typing import Any, Dict, Optional, List
from typing_extensions import List

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF_FACTOR = 1

class EinsteinLLMModel(LLM):
    """A custom LLM class that uses the Einstein LLM Gateway API."""
    
    gateway_server: str = os.environ.get("EINSTEIN_GATEWAY_SERVER")
    gateway_path: str = "v1.0"
    api_key: str = os.environ.get("EINSTEIN_API_KEY")
    model_name: str = "llmgateway__OpenAIGPT4_32k"
    x_client_feature_id: str = "EinsteinDocsAnswers"
    x_sfdc_app_context: str = "EinsteinGPT"
    x_sfdc_core_tenant_id: str = "core/prod1/00DDu0000008cuqMAA"

    def _build_headers(self) -> Dict[str, str]:
        """Helper to build headers, filtering out None values."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"API_KEY {self.api_key}",
            "x-client-feature-id": self.x_client_feature_id,
            "x-sfdc-app-context": self.x_sfdc_app_context,
            "x-sfdc-core-tenant-id": self.x_sfdc_core_tenant_id
        }
        # Return only headers with non-None values
        return {k: v for k, v in headers.items() if v is not None}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        headers = self._build_headers()
        data = {
            "model": self.model_name,
            "prompt": prompt
        }
        payload = {k: v for k, v in data.items() if v is not None}
        api_url= f"https://{self.gateway_server}/{self.gateway_path}/generations"
        last_exception = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                logging.debug(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: POST {api_url}")
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=kwargs.get("request_timeout", 60)
                )

                response.raise_for_status()
                try:
                    json_response = response.json()
                    generations = json_response.get("generations")
                    if isinstance(generations, list) and len(generations) > 0:
                         first_gen = generations[0]
                         if isinstance(first_gen, dict):
                              text_result = first_gen.get("text")
                              if isinstance(text_result, str):
                                   logging.debug(f"Attempt {attempt + 1}: Success.")
                                   return text_result # Successfully extracted text

                    logging.error(f"Attempt {attempt + 1}: Unexpected JSON structure in successful response: {json_response}")
                    raise ValueError(f"Invalid response structure received from API: {json_response}")

                except json.JSONDecodeError as e:
                    logging.error(f"Attempt {attempt + 1}: Failed to decode JSON from successful response: {e}. Response text: {response.text}")
                    raise ValueError(f"Invalid JSON received from API: {e}") from e
                # --- End Success Path ---

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429 and attempt < MAX_RETRIES:
                    base_delay = INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    retry_after = e.response.headers.get("Retry-After")
                    wait_time = base_delay
                    if retry_after:
                        try:
                            wait_time = max(base_delay, float(retry_after))
                            logging.info(f"Using Retry-After header value: {wait_time:.2f}s")
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid Retry-After header value: '{retry_after}'. Using calculated backoff.")

                    logging.warning(
                        f"Attempt {attempt + 1}: Received HTTP 429 (Too Many Requests). "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    last_exception = e # Store the exception
                    time.sleep(wait_time)
                    continue
                else:
                    # Handle other HTTP errors or the final 429 error
                    status_code = e.response.status_code if e.response is not None else "Unknown"
                    response_text = e.response.text if e.response is not None else "No response body"
                    logging.error(f"Attempt {attempt + 1}: HTTP Error {status_code}: {response_text}")
                    last_exception = ValueError(f"API call failed with HTTP status {status_code}: {response_text}")
                    break # Exit loop for non-retryable HTTP errors or final 429

            except requests.exceptions.Timeout as e:
                logging.error(f"Attempt {attempt + 1}: Request timed out: {e}")
                last_exception = ConnectionError(f"API request timed out after {attempt + 1} attempt(s)")
                break # Exit loop on timeout (usually not retried unless idempotent)
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Attempt {attempt + 1}: Connection Error: {e}")
                last_exception = ConnectionError(f"Failed to connect to API after {attempt + 1} attempt(s)")
                # Potentially add retry logic here too if desired
                break # Exit loop on connection error
            except requests.exceptions.RequestException as e:
                # Catch other requests library errors (e.g., invalid URL)
                logging.error(f"Attempt {attempt + 1}: Request Exception: {e}")
                last_exception = ValueError(f"Error during API call setup or execution after {attempt + 1} attempt(s)")
                break # Exit loop

        # If the loop completes without returning (meaning all retries failed or a non-retryable error occurred)
        logging.error(f"API call failed permanently after {attempt + 1} attempts.")
        if last_exception:
            raise last_exception # Re-raise the last significant exception captured
        else:
            # This case should ideally not be reached if exceptions are handled properly
            raise ConnectionError("API call failed after multiple attempts for an unknown reason.")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "url": f"https://{self.gateway_server}/{self.gateway_path}/generations",
            "x_client_feature_id": self.x_client_feature_id,
            "x_sfdc_app_context": self.x_sfdc_app_context,
            "x_sfdc_core_tenant_id": self.x_sfdc_core_tenant_id,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model."""
        return "einstein_llm"