import logging
import json
import os 
from firebase_admin import credentials, firestore, initialize_app
from firebase_admin.exceptions import FirebaseError
import uuid 
from typing import TypedDict, Annotated, List, Union, Any, Dict, Optional

logger = logging.getLogger(__name__)


_db = None
_app = None


STATIC_APP_ID = "ai-support-agent-app" 

def _initialize_firestore():
    """
    Initializes the Firebase Admin SDK and Firestore client.
    This function should be called once during application startup.
    """
    global _db, _app

    if _app is None:
        try:
            
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                _app = initialize_app(cred)
                logger.info("Firebase initialized using service account credentials.")
            else:
                
                try:
                    # Try ApplicationDefault which looks for credentials in the environment
                    cred = credentials.ApplicationDefault()
                    _app = initialize_app(cred)
                    logger.info("Firebase initialized using Application Default Credentials.")
                except Exception as e:
                    logger.warning(f"Application Default Credentials failed ({e}). Attempting initialization without explicit credentials. This might require default project setup or service account.", exc_info=True)
                    # Fallback to initializing without explicit credentials (might work in some environments)
                    _app = initialize_app() # This will attempt to find project ID from environment

            _db = firestore.client()
            logger.info("Firebase Admin SDK and Firestore client initialized successfully.")
        except ValueError as ve:
            logger.error(f"Firebase initialization error (ValueError): {ve}. Check if Firebase is already initialized or if projectId is missing/invalid.", exc_info=True)
            _app = None # Reset app to None if initialization failed
            _db = None
        except FirebaseError as fe:
            logger.error(f"Firebase initialization error (FirebaseError): {fe}. Check credentials or project configuration.", exc_info=True)
            _app = None # Reset app to None if initialization failed
            _db = None
        except Exception as e:
            logger.error(f"Unexpected error during Firebase initialization: {e}", exc_info=True)
            _app = None # Reset app to None if initialization failed
            _db = None

def store_feedback(feedback_data: Dict[str, Any]):
    """
    Stores user feedback in Firestore.
    Args:
        feedback_data: A dictionary containing feedback details
                       (e.g., session_id, message_content, feedback_type, comment).
    """
    global _db

    if _db is None:
        _initialize_firestore() # Attempt to initialize if not already
        if _db is None: # If initialization still failed, cannot proceed
            logger.error("Firestore client not initialized. Cannot store feedback.")
            raise Exception("Firestore not available to store feedback.")

    try:
        # Use the static app ID for the collection path
        app_id = STATIC_APP_ID

        
        user_id = feedback_data.get('session_id', str(uuid.uuid4())) # Fallback to new UUID if session_id is missing

        
        collection_path = f"artifacts/{app_id}/public/data/feedback"

        # Add a new document to the 'feedback' collection
        doc_ref = _db.collection(collection_path).document() # Firestore generates a unique ID
        
        # Add a timestamp to the feedback data
        feedback_data['timestamp'] = firestore.SERVER_TIMESTAMP # Use server timestamp for consistency

        doc_ref.set(feedback_data)
        logger.info(f"Feedback stored successfully with ID: {doc_ref.id} in collection: {collection_path}")
    except FirebaseError as fe:
        logger.error(f"Firestore operation failed: {fe}", exc_info=True)
        raise Exception(f"Failed to store feedback in Firestore: {fe}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while storing feedback: {e}", exc_info=True)
        raise Exception(f"An unexpected error occurred: {e}")


_initialize_firestore()
