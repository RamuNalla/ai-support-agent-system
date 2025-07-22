import logging
import os
import faiss 
import numpy as np 
from typing import List, Dict, Any
from langchain_core.documents import Document 

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, index_path: str = "faiss_index.bin"):
        self.index_path = index_path            # file path where the FAISS index is stored
        self.index = None                       # This will hold the actual FAISS index object (faiss.IndexFlatL2)
        self.doc_store = {}                     # doc_store (Python dictionary) to store the actual text content and metadata. FAISS itself only stores numerical vectors, so we need to manage the associated data separately.
        logger.info(f"FAISS VectorDBManager initialized with index path: {self.index_path}")



    def load_or_create_index(self, vector_size: int = 384):             # This function typically called by the agent when agent starts up. Loads an existing FAISS index from disk if the file exists, or creates a new empty index if the file is not found

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)          # faiss.read_index loads the FAISS index from the specified file.
                logger.info(f"FAISS index loaded from {self.index_path}. Index size: {self.index.ntotal} vectors.")

                if not self.doc_store and self.index.ntotal > 0:
                    logger.warning("Doc store not loaded during index load. Retrieval will only work if doc_store is populated later (e.g., by ingestion).")

            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                self._create_empty_index(vector_size)                   # If loading fails (e.g., corrupted file), create a new empty index to proceed.
        else:
            logger.info(f"FAISS index not found at {self.index_path}. Creating a new empty index.")
            self._create_empty_index(vector_size)



    def _create_empty_index(self, vector_size: int):            # Helper method to create a new, empty FAISS index.
        self.index = faiss.IndexFlatL2(vector_size)             # IndexFlatL2 is a simple index that performs a brute-force (linear scan) search for the closest vectors based on L2 (Euclidean) distance
        self.doc_store = {}                                     # Reset doc store when creating a new index
        logger.info(f"New empty FAISS index created with vector size {vector_size}.")

    


    def upsert_vectors(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):           # Inserts vectors and their associated payloads into the FAISS index.

        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call load_or_create_index or _create_empty_index first.")

        vectors_np = np.array(vectors).astype('float32')        # Convert the list of vectors into a NumPy array of float32, which FAISS requires.
        self.index.add(vectors_np)                              # Add the vectors to the FAISS index. FAISS assigns an internal integer ID to each vector.

        current_total = self.index.ntotal
        for i, payload in enumerate(payloads):  
            faiss_internal_id = str(current_total - len(payloads) + i)      # We map FAISS's internal integer IDs (which are sequential) to our payloads.
            self.doc_store[faiss_internal_id] = payload                     # The internal ID for the first newly added vector will be self.index.ntotal - len(payloads).

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

        query_vector_np = np.array([query_vector]).astype('float32')        # Convert the single query vector into a 2D NumPy array of float32, as FAISS expects.
        distances, indices = self.index.search(query_vector_np, limit)      # Perform the search. 'distances' contains the L2 distances (lower is more similar). 'indices' contains the internal FAISS integer IDs of the found vectors.

        results = []
        
        for i, idx in enumerate(indices[0]):                                # Iterate through the search results. indices[0] because we searched with one query vector.
            if idx == -1:                                                   # FAISS returns -1 for slots if 'limit' is greater than total vectors
                continue
        
            payload = self.doc_store.get(str(idx))                          # Lookup in our doc_store
            if payload:                                                     # Ensure payload exists
                results.append({"payload": payload, "score": float(distances[0][i])})
        
        logger.info(f"Found {len(results)} search results in FAISS index.")
        return results




    def save_index(self):                                                       # Saves the current FAISS index to disk as a binary file. This is crucial for persistence so the index doesn't have to be rebuilt every time.
    
        if self.index:
            try:
                faiss.write_index(self.index, self.index_path)                  # faiss.write_index saves the index structure and vectors.
                logger.info(f"FAISS index saved to {self.index_path}.")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("No FAISS index to save. Index is None.")

