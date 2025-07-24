import requests
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AgentAPIClient:           # Client for interacting with the AI Support Agent FastAPI backend.

    def __init__(self, base_url: str):
        self.base_url = base_url
        logger.info(f"AgentAPIClient initialized with base URL: {self.base_url}")

    def chat(self, message: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:     # Sends a chat message to the agent and retrieves its response.
        """
        Sends a chat message to the agent and retrieves its response.

        Args:
            message (str): The current message from the user.
            chat_history (List[Dict[str, Any]]): A list of previous conversation turns.
                                                 Each item should be a dict like
                                                 {"type": "human", "content": "..."} or
                                                 {"type": "ai", "content": "..."}.

        Returns:
            Dict[str, Any]: The agent's response, including the updated chat history
                            and potentially a clarifying question.
                            Example: {"response": "...", "chat_history": [...], "clarifying_question": null}
        """
        endpoint = f"{self.base_url}/api/v1/chat"               # construct a full URL
        payload = {                                             # JSON body that is sent in the POST request
            "message": message,
            "chat_history": chat_history
        }
        headers = {"Content-Type": "application/json"}          # Informing rhe server that the request body is in JSON format

        logger.info(f"Sending chat request to {endpoint} with message: '{message[:50]}...'")
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=120)   # Increased timeout (60 seconds for the server to respond)
            response.raise_for_status()                                                     # Raise an HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            logger.info(f"Received chat response: {response_data.get('response', '')[:50]}...")
            return response_data                                                            # return the response dictionary to the caller (app.py)
        
        except requests.exceptions.Timeout:
            logger.error(f"Request to {endpoint} timed out after 60 seconds.")
            return {"response": "The agent took too long to respond. Please try again.", "chat_history": chat_history, "clarifying_question": None}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with agent service at {endpoint}: {e}", exc_info=True)
            error_detail = "An unknown error occurred."
            if e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("detail", error_detail)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            return {"response": f"Error: Could not connect to the AI agent. Details: {error_detail}", "chat_history": chat_history, "clarifying_question": None}
        
        except Exception as e:
            logger.error(f"An unexpected error occurred in AgentAPIClient.chat: {e}", exc_info=True)
            return {"response": f"An unexpected error occurred: {e}", "chat_history": chat_history, "clarifying_question": None}

