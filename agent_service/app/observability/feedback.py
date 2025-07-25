import logging
import json
from datetime import datetime
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

FEEDBACK_FILE_PATH = os.getenv("FEEDBACK_FILE_PATH", "feedback.jsonl")      # file path for storing feedback.

def store_feedback(feedback_data: Dict[str, Any]):                          # Appends a new feedback entry to a file. Each entry is a JSON line format.

    logger.info("Storing user feedback...")
    try:
        feedback_data["timestamp"] = datetime.now().isoformat()             # Add a timestamp to the feedback data for when it was received
        
        with open(FEEDBACK_FILE_PATH, 'a') as f:                            # Open the file in append mode ('a') and write the JSON object followed by a newline
            f.write(json.dumps(feedback_data) + '\n')
        
        logger.info(f"Feedback stored successfully to {FEEDBACK_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to store feedback to file: {e}", exc_info=True)
        raise                                                               # Re-raise to let the API endpoint handle the error