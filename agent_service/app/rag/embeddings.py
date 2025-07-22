import logging
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):           # Creates 384 dimension vectors

        self.model_name = model_name
        try:
            self.embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)
            logger.info(f"Embeddings model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embeddings model '{self.model_name}': {e}", exc_info=True)
            raise 

    def get_embeddings_model(self):             # Simple method to return the initialized embeddings model instance.
        return self.embeddings

