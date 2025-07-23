import os
import logging
from langchain_community.document_loaders import TextLoader, DirectoryLoader    # For loading documents
from langchain.text_splitter import RecursiveCharacterTextSplitter              # For splitting documents
from app.rag.embeddings import EmbeddingsGenerator                # Your custom module for embeddings
from app.rag.vector_db import VectorDBManager                     # Your custom module for FAISS
from uuid import uuid4                                                          # To generate unique IDs for document chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCS_PATH = "data/knowledge_base/docs"            # Path to your knowledge base documents 
FAISS_INDEX_FILE = "faiss_index.bin"              # FAISS settings: This is where the generated FAISS index will be saved.
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"                      # Embeddings model name (must match the one used in the agent's core.py)
CHUNK_SIZE = 1000                                               # Max characters per chunk
CHUNK_OVERLAP = 200                                             # Overlap between consecutive chunks to maintain context

def ingest_documents():             # Loads documents from the specified path, splits them into chunks, generates embeddings for each chunk, and upserts them into the FAISS index, then saves the index to disk for persistence.
    
    logger.info(f"Starting document ingestion from: {DOCS_PATH}")

    try:
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.md", loader_cls=lambda x: TextLoader(x, encoding='utf-8'))      # DirectoryLoader can load multiple markdown (.md) files from a directory.
        documents = loader.load()                                                       # Loads documents into a list of LangChain Document objects
        logger.info(f"Loaded {len(documents)} documents.")
    except Exception as e:
        logger.error(f"Error loading documents from {DOCS_PATH}: {e}", exc_info=True)
        return

    if not documents:
        logger.warning("No documents found to ingest. Please ensure files exist in DOCS_PATH and glob pattern is correct.")
        return

    text_splitter = RecursiveCharacterTextSplitter(                         # Split documents into chunks. RecursiveCharacterTextSplitter tries to split text intelligently (e.g., by paragraphs, then sentences) to preserve semantic meaning within chunks.
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,                                    # Use character count for chunk size
        is_separator_regex=False,                               # Treat separators as literal strings
    )
    chunks = text_splitter.split_documents(documents)           # Splits loaded documents into smaller chunks
    logger.info(f"Split documents into {len(chunks)} chunks.")

    embeddings_generator = EmbeddingsGenerator(model_name=EMBEDDINGS_MODEL_NAME)        # Generate embeddings for each chunk
    embeddings_model = embeddings_generator.get_embeddings_model()

    chunk_contents = [chunk.page_content for chunk in chunks]               # Extract just the text content from each chunk to pass to the embedding model
    try:
        vectors = embeddings_model.embed_documents(chunk_contents)          # embed_documents takes a list of strings and returns a list of corresponding embeddings (list of floats).
        logger.info(f"Generated embeddings for {len(vectors)} chunks.")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return

    ids = [str(uuid4()) for _ in chunks]            # Generate a unique UUID for each chunk. FAISS uses internal integer IDs, but we'll generate UUIDs for logical tracking. The VectorDBManager will handle the mapping.
    payloads = []
    for i, chunk in enumerate(chunks):
        payloads.append({
            "source": chunk.metadata.get("source", "unknown"),          # Original file path
            "chunk_id": i,                                              # Index of the chunk within its original document
            "content": chunk.page_content                               # The actual text content of the chunk
        })


    vector_db_manager = VectorDBManager(index_path=FAISS_INDEX_FILE)    # Initialize FAISS VectorDBManager and upsert data
    vector_db_manager._create_empty_index(vector_size=embeddings_model.client.get_sentence_embedding_dimension())
    vector_db_manager.upsert_vectors(ids=ids, vectors=vectors, payloads=payloads)

    logger.info(f"FAISS index now contains {vector_db_manager.index.ntotal} vectors.")

    vector_db_manager.save_index()               # Save the FAISS index to disk If you don't save, the index will be lost when the script finishes.

    logger.info("Document ingestion completed and FAISS index saved.")

if __name__ == "__main__":
    ingest_documents()                          # ingest_documents() is called only when the script is executed directly.

