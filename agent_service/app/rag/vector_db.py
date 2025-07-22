import logging
import os
import faiss 
import numpy as np 
from typing import List, Dict, Any
from langchain_core.documents import Document 

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, index_path: str = "faiss_index.bin"):
        self.index_path = index_path            # Path to the binary file where the FAISS index is stored
        self.index = None                       # This will hold the FAISS index object (e.g., IndexFlatL2)
        self.doc_store = {}                     # doc_store is a simple Python dictionary to store the actual text content and metadata. FAISS itself only stores numerical vectors, so we need to manage the associated data separately.
        logger.info(f"FAISS VectorDBManager initialized with index path: {self.index_path}")

    def load_or_create_index(self, vector_size: int = 384):
        """
        Loads an existing FAISS index from disk if the file exists,
        or creates a new empty index if the file is not found or loading fails.
        This method is typically called when the agent service starts up.
        """
        if os.path.exists(self.index_path):
            try:
                # faiss.read_index loads the FAISS index from the specified binary file.
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded from {self.index_path}. Index size: {self.index.ntotal} vectors.")

                # IMPORTANT: For a production-ready FAISS setup, you would also need to
                # save and load 'self.doc_store' (the payloads) alongside the index.
                # This example simplifies by assuming doc_store is rebuilt during ingestion
                # or that the agent service is short-lived.
                # A common approach is to save doc_store as a JSON or pickle file.
                if not self.doc_store and self.index.ntotal > 0:
                    logger.warning("Doc store not loaded during index load. Retrieval will only work if doc_store is populated later (e.g., by ingestion).")

            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                # If loading fails (e.g., corrupted file), create a new empty index to proceed.
                self._create_empty_index(vector_size)
        else:
            logger.info(f"FAISS index not found at {self.index_path}. Creating a new empty index.")
            self._create_empty_index(vector_size)

    def _create_empty_index(self, vector_size: int):
        """
        Helper method to create a new, empty FAISS index.
        `IndexFlatL2` is a basic index that performs a brute-force search using L2 (Euclidean) distance.
        It's suitable for smaller to medium datasets. For very large datasets, more advanced
        indexes like `IndexIVFFlat` or `IndexHNSW` would be used for performance.
        """
        self.index = faiss.IndexFlatL2(vector_size)
        self.doc_store = {} # Reset doc store when creating a new index
        logger.info(f"New empty FAISS index created with vector size {vector_size}.")

    def upsert_vectors(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """
        Inserts vectors and their associated payloads into the FAISS index.
        Note: FAISS primarily supports adding vectors. To "update" an existing vector,
        you typically add a new one and manage duplicates outside FAISS, or rebuild the index.
        For simplicity, this method assumes new data is being appended.

        :param ids: List of unique string IDs for the vectors (used for logical mapping).
        :param vectors: List of vector embeddings (e.g., from EmbeddingsGenerator).
        :param payloads: List of dictionaries containing metadata for each vector (e.g., content, source).
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call load_or_create_index or _create_empty_index first.")

        # Convert the list of vectors into a NumPy array of float32, which FAISS requires.
        vectors_np = np.array(vectors).astype('float32')

        # Add the vectors to the FAISS index. FAISS assigns an internal integer ID to each vector.
        self.index.add(vectors_np)

        # Store payloads in the doc_store dictionary.
        # We map FAISS's internal integer IDs (which are sequential) to our payloads.
        # The internal ID for the first newly added vector will be self.index.ntotal - len(payloads).
        # We use string keys for the dictionary for consistency in lookup.
        current_total = self.index.ntotal
        for i, payload in enumerate(payloads):
            faiss_internal_id = str(current_total - len(payloads) + i)
            self.doc_store[faiss_internal_id] = payload

        logger.info(f"Upserted {len(ids)} vectors into FAISS index (total: {self.index.ntotal}).")

    def search_vectors(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the most similar vectors to the provided query vector.
        This is the core retrieval operation for RAG.

        :param query_vector: The embedding of the query text (e.g., from EmbeddingsGenerator).
        :param limit: The maximum number of similar results to return.
        :return: A list of dictionaries, each containing 'payload' (document metadata) and 'score'.
        """
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot perform search.")
            return []

        # Convert the single query vector into a 2D NumPy array of float32, as FAISS expects.
        query_vector_np = np.array([query_vector]).astype('float32')

        # Perform the search.
        # 'distances' contains the L2 distances (lower is more similar).
        # 'indices' contains the internal FAISS integer IDs of the found vectors.
        distances, indices = self.index.search(query_vector_np, limit)

        results = []
        # Iterate through the search results. indices[0] because we searched with one query vector.
        for i, idx in enumerate(indices[0]):
            if idx == -1: # FAISS returns -1 for slots if 'limit' is greater than total vectors
                continue
            # Retrieve the full payload (text content and metadata) using the internal FAISS ID.
            payload = self.doc_store.get(str(idx)) # Lookup in our doc_store
            if payload: # Ensure payload exists
                results.append({"payload": payload, "score": float(distances[0][i])})
        
        logger.info(f"Found {len(results)} search results in FAISS index.")
        return results

    def save_index(self):
        """
        Saves the current FAISS index to disk as a binary file.
        This is crucial for persistence so the index doesn't have to be rebuilt every time.
        """
        if self.index:
            try:
                # faiss.write_index saves the index structure and vectors.
                faiss.write_index(self.index, self.index_path)
                # IMPORTANT: For a complete persistence solution, you would also need to
                # save 'self.doc_store' (the payloads) to a separate file (e.g., JSON or pickle)
                # and load it alongside the FAISS index. This example simplifies by only saving the index.
                logger.info(f"FAISS index saved to {self.index_path}.")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("No FAISS index to save. Index is None.")

