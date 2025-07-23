import logging
import os
import faiss
import numpy as np
import pickle # For serializing/deserializing the doc_store
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, index_path: str):                # Initializes the FAISS VectorDBManager.

        self.index_path = index_path
        self.index: Optional[faiss.Index] = None
        self.doc_store: Dict[str, Any] = {}
        self.load_index() 
        logger.info(f"FAISS VectorDBManager initialized with index path: {self.index_path}")


    def _create_empty_index(self, vector_size: int):        # Helper method to create a new, empty FAISS HNSW index.
        M = 16                                              # M: The number of neighbors for each vector in the HNSW graph.
        self.index = faiss.IndexHNSWFlat(vector_size, M)    
        self.doc_store = {}                                 # Reset doc_store for a new index
        logger.info(f"New empty FAISS HNSW index created with M={M} and vector size {vector_size}.")


    def load_index(self):                                   # Loads an existing FAISS index and its associated doc_store from disk.
        
        docstore_path = self.index_path.replace('.bin', '_docstore.pkl')
        
        if os.path.exists(self.index_path) and os.path.exists(docstore_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(docstore_path, 'rb') as f:
                    self.doc_store = pickle.load(f)
                logger.info(f"Loaded existing FAISS index from {self.index_path} with {self.index.ntotal} vectors.")
                logger.info(f"Loaded doc_store from {docstore_path} with {len(self.doc_store)} entries.")
            except Exception as e:
                logger.error(f"Error loading FAISS index or doc store: {e}", exc_info=True)
                self.index = None
                self.doc_store = {}
        else:
            logger.warning(f"FAISS index or doc store file not found at {self.index_path}. Index will be created on first upsert or remain uninitialized.")
            self.index = None
            self.doc_store = {}


    def upsert_vectors(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):       # Inserts vectors and their associated payloads into the FAISS index.

        if not self.index:    # If index was not loaded (e.g., first run of ingest_data.py), create a new one.
            if not vectors:
                logger.error("No vectors provided to upsert, cannot create a new index.")
                return
            vector_size = len(vectors[0])
            self._create_empty_index(vector_size)

        vectors_np = np.array(vectors).astype('float32')

        start_id = self.index.ntotal 
        for i, payload in enumerate(payloads):
            faiss_internal_id = str(start_id + i)               # Convert to string for dictionary key
            self.doc_store[faiss_internal_id] = payload

        self.index.add(vectors_np)                              # Add vectors to the FAISS index. This updates self.index.ntotal.
        
        logger.info(f"Upserted {len(vectors)} vectors. Total vectors in index: {self.index.ntotal}.")


    def search_vectors(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:        # Searches the FAISS index for the most similar vectors to the query vector.

        if not self.index:
            logger.warning("Search called, but FAISS index is not initialized. Returning empty results.")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Search called, but FAISS index is empty. Returning empty results.")
            return []

        query_vector_np = np.array([query_vector]).astype('float32')
        
        distances, faiss_ids = self.index.search(query_vector_np, limit)
        
        results = []
    
        for i in range(len(distances[0])):          # Iterate through the search results (distances[0] and faiss_ids[0] because we queried with one vector)
            faiss_id = str(faiss_ids[0][i])         # Convert internal ID to string for doc_store lookup
            distance = distances[0][i]
            
            payload = self.doc_store.get(faiss_id)  # Retrieve the full payload using the FAISS internal ID
            if payload:
                payload['score'] = float(distance) 
                results.append(payload)
            else:
                logger.warning(f"Payload not found for FAISS ID: {faiss_id}. This might indicate a mismatch between index and doc_store.")

        logger.info(f"Found {len(results)} search results in FAISS index.")
        return results



    def save_index(self):                   # Saves the current FAISS index and its associated doc_store to disk.
        if self.index:
            try:
                faiss.write_index(self.index, self.index_path)
                docstore_path = self.index_path.replace('.bin', '_docstore.pkl')            # Save the doc_store (payloads) using pickle
                with open(docstore_path, 'wb') as f:
                    pickle.dump(self.doc_store, f)
                logger.info(f"FAISS index and doc store saved to {self.index_path} and {docstore_path}.")
            except Exception as e:
                logger.error(f"Error saving FAISS index or doc store: {e}", exc_info=True)
        else:
            logger.warning("No FAISS index to save. Index is None.")

