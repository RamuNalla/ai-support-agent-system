import logging
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document           
from app.rag.embeddings import EmbeddingsGenerator      # custom module for embeddings
from app.rag.vector_db import VectorDBManager           # custom module for FAISS
from app.config.settings import settings                # settings to get FAISS_INDEX_PATH

logger = logging.getLogger(__name__)            # Initialize logger

class AgentState(TypedDict):                                        # Define the state of our graph
    messages: Annotated[list, lambda left, right: left + right]     # List of messages (HumanMessage, AIMessage)
    relevant_docs: List[Document]                                   # New field to store retrieved documents (LangChain Document objects)
class Agent:                                    # Initializes the agent with the Gemini LLM.
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not provided.")

        logger.info("Initializing Gemini LLM")
        try:
            self.llm = ChatGoogleGenerativeAI(              # Initialize ChatGoogleGenerativeAI with the provided API key
                model="gemini-1.5-flash",                   # Using gemini-flash for general text generation
                google_api_key=gemini_api_key
            )
            logger.info("Gemini LLM initialized.")
            
            self.embeddings_generator = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")      # Initialize Embeddings Generator.
            self.embeddings_model = self.embeddings_generator.get_embeddings_model()
            logger.info("Embeddings model initialized.")

            self.vector_db_manager = VectorDBManager(index_path=settings.FAISS_INDEX_PATH)                                                  # Initialize FAISS Vector DB Manager
            #self.vector_db_manager.load_or_create_index(vector_size=self.embeddings_model.client.get_sentence_embedding_dimension())        # Load the FAISS index.
            logger.info("FAISS Vector DB Manager initialized and index loaded.")

        except Exception as e:
            logger.error(f"Error initializing Agent components: {e}", exc_info=True)
            raise                                           # Re-raise the exception to prevent the app from starting incorrectly


    def retrieve_documents(self, state: AgentState) -> AgentState:          # Langgraph node to retrive relevant documents from the FAISS index

        latest_human_message = None
        for msg in reversed(state['messages']):                 # Find the most recent human message in the conversation history.
            if isinstance(msg, HumanMessage):
                latest_human_message = msg.content
                break

        if not latest_human_message:
            logger.warning("No human message found for retrieval. Skipping retrieval.")
            return {"relevant_docs": []}                        # If no human message, return an empty list of documents.

        logger.info(f"Retrieving documents for query: '{latest_human_message[:50]}...'")
        
        try:
            query_vector = self.embeddings_model.embed_query(latest_human_message)          # Generate an embedding for the user's query.
            search_results = self.vector_db_manager.search_vectors(query_vector, limit=5)   # Search the FAISS index for the top 3 most similar documents.

            relevant_docs = []
            for res in search_results:                          # Convert the raw search results (payloads) into LangChain Document objects.  This makes the retrieved information consistent and easy to pass to the LLM.
                content = res.get('content', '')
                source = res.get('source', 'unknown')
                score = res['score']
                relevant_docs.append(Document(page_content=content, metadata={"source": source, "score": score}))

            logger.info(f"Retrieved {len(relevant_docs)} documents.")
            return {"relevant_docs": relevant_docs}                      # Return the retrieved documents to update the 'relevant_docs' channel in the AgentState.
        
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            return {"relevant_docs": []}                                 # Return empty list on error        



    def generate_response_with_rag(self, state: AgentState) -> AgentState:   # Langgraph node to invoke LLM and get the response

        messages = state['messages']                    # Current conversation history
        relevant_docs = state['relevant_docs']          # Documents retrieved in the previous step

        context_str = ""
        if relevant_docs:
            context_str = "\n\nRelevant Context:\n"
            for i, doc in enumerate(relevant_docs):             # Format each retrieved document for inclusion in the prompt. Including source and score can be helpful for debugging and future UI features.
                context_str += f"--- Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}, Score: {doc.metadata.get('score', 'N/A'):.2f}) ---\n"
                context_str += doc.page_content + "\n"
            context_str += "\n"
            logger.info(f"Adding {len(relevant_docs)} documents to LLM context.")

        system_prompt = (                   # Create a system message that explicitly instructs the LLM on its role and how to use the context. This is crucial for RAG effectiveness: it guides the LLM to use the provided information.
            "You are a helpful AI support agent that answers questions about the provided information. "
            "Use the relevant context provided below to answer the user's question accurately and concisely. "
            "If the question cannot be answered from the provided context, state that you don't have enough information "
            "and suggest contacting a human agent.\n\n"
            f"{context_str}"                # Inject the retrieved context here
        )

        llm_messages = [SystemMessage(content=system_prompt)] + messages        # Construct the full list of messages for the LLM. The system message with context comes first, followed by the actual conversation history. This ensures the LLM always has the context available at the beginning of its input.

        logger.info(f"Calling LLM with RAG context and messages: {llm_messages}")
        try:
            response = self.llm.invoke(llm_messages)                # Invoke the LLM with the augmented messages.
            logger.info(f"LLM response received with RAG: {response.content[:100]}...")
            return {"messages": [response]}                         # Return the AI's response to update the 'messages' channel in the AgentState.
        
        except Exception as e:                                      # If the LLM call fails, return an error message as an AI response.
            logger.error(f"Error calling LLM with RAG: {e}", exc_info=True)
            return {"messages": [AIMessage(content=f"Error: Could not get a response from the AI with RAG. {e}")]}



    def build_graph(self):                                  # Graph building

        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("generate_response_with_rag", self.generate_response_with_rag)        
        
        workflow.set_entry_point("retrieve_documents")                                  # Set 'retrieve_documents' as the entry point of the graph
        
        workflow.add_edge("retrieve_documents", "generate_response_with_rag")
        workflow.add_edge("generate_response_with_rag", END)                            # Define an edge from 'call_llm' to END, meaning the graph finishes after the LLM call

        app = workflow.compile()                            # Compile the workflow into a runnable LangGraph application
        logger.info("LangGraph workflow compiled.")
        return app

