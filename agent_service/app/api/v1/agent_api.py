import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from typing import Dict, Any, List, Optional
from app.agent.core import Agent, AgentState
from app.config.settings import settings
from app.observability.feedback import store_feedback
from app.observability.metrics import (             # Import defined metrics
    CHAT_REQUESTS_TOTAL,
    CHAT_ERRORS_TOTAL,
    CHAT_LATENCY_HISTOGRAM,
    ACTIVE_CHAT_REQUESTS_GAUGE                    
)

logger = logging.getLogger(__name__)                # Initialize logger

router = APIRouter()                                # Create an API router for agent-related endpoints (helps organizing endpoints)

agent_instance: Agent = None                        # Global agent instance

def get_agent() -> Agent:                           # Dependency to get the agent instance. Is initalized only once when the application starts up
    global agent_instance
    if agent_instance is None:
        try:
            agent_instance = Agent(gemini_api_key=settings.GEMINI_API_KEY)
            logger.info("Agent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {e}")
    return agent_instance

class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, Any]]
    clarifying_question: Optional[str] = None 
    relevant_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="List of relevant documents (sources) found during RAG.")

class FeedbackRequest(BaseModel): 
    session_id: str
    message_content: str
    feedback_type: str              # "positive" or "negative"
    comment: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, agent: Agent = Depends(get_agent)):
    logger.info(f"Received chat request: {request.message}")
    ACTIVE_CHAT_REQUESTS_GAUGE.inc()             # Increment when request starts
    start_time = time.time()                # Record start time

    try:
        langchain_chat_history = []                             # Reconstruct chat history from request
        for msg in request.chat_history:
            if msg["type"] == "human":
                langchain_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                ai_message_kwargs = {"content": msg["content"]}
                if "tool_calls" in msg and msg["tool_calls"]:
                    ai_message_kwargs["tool_calls"] = msg["tool_calls"]
                langchain_chat_history.append(AIMessage(**ai_message_kwargs))
            elif msg["type"] == "tool":
                langchain_chat_history.append(ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"]))
            elif msg["type"] == "system":
                langchain_chat_history.append(SystemMessage(content=msg["content"]))

        
        initial_state = AgentState(messages=langchain_chat_history + [HumanMessage(content=request.message)],       # Create the initial state for the graph
                                   relevant_docs=[],
                                   tool_calls=[],
                                   tool_output=None,
                                   clarifying_question=None) 
                                    
        # Run the agent's graph
        final_state: AgentState = await agent.build_graph().ainvoke(initial_state)
        
        # Now, final_state should be the full AgentState TypedDict
        logger.debug(f"Final agent state: {final_state}")

        if final_state is None:
            logger.error("Agent graph stream returned no final state. This should not happen if the graph compiled.")
            CHAT_ERRORS_TOTAL.labels(error_type="empty_response").inc()
            raise HTTPException(status_code=500, detail="Agent returned an empty response (no final state).")
        
        # Check if 'messages' key exists and is not empty
        if 'messages' not in final_state:
            logger.error(f"Final state from agent graph is missing 'messages' key. Final state type: {type(final_state)}, Content: {final_state}")
            CHAT_ERRORS_TOTAL.labels(error_type="invalid_state").inc()
            raise HTTPException(status_code=500, detail="Agent returned an invalid state (missing messages key).")

        if not final_state['messages']:
            logger.error(f"Final state 'messages' list is empty. Final state type: {type(final_state)}, Content: {final_state}")
            CHAT_ERRORS_TOTAL.labels(error_type="empty_messages_list").inc()
            raise HTTPException(status_code=500, detail="Agent returned an empty messages list.")

        clarifying_q = final_state.get('clarifying_question')        # Check for clarifying question first 
        if clarifying_q:
            logger.info(f"Agent asked a clarifying question: '{clarifying_q}'")
            CHAT_REQUESTS_TOTAL.labels(status="clarify").inc()          # Increment for clarifying question
            CHAT_LATENCY_HISTOGRAM.observe(time.time() - start_time)
            return ChatResponse(response="", chat_history=[], clarifying_question=clarifying_q, relevant_docs=[])     # Return the clarifying question directly to the user


        final_ai_response = None
        for msg in reversed(final_state['messages']):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_ai_response = msg.content                             # find the final AI response if no clarification 
                break
            elif isinstance(msg, HumanMessage): 
                 if not final_ai_response and isinstance(final_state['messages'][-1], AIMessage):       
                    final_ai_response = final_state['messages'][-1].content
                 break 

        if not final_ai_response:
            if isinstance(final_state['messages'][-1], AIMessage):
                final_ai_response = final_state['messages'][-1].content
            else:
                final_ai_response = "I processed your request, but I couldn't formulate a direct answer. Please check the logs for details."
            CHAT_ERRORS_TOTAL.labels(error_type="no_final_ai_response").inc()

        CHAT_REQUESTS_TOTAL.labels(status="success").inc()
        CHAT_LATENCY_HISTOGRAM.observe(time.time() - start_time)
        
        updated_chat_history = []                       # Update chat history for the response
        for msg in final_state['messages']:
            if isinstance(msg, HumanMessage):
                updated_chat_history.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                updated_msg = {"type": "ai", "content": msg.content}
                if msg.tool_calls:
                    updated_msg["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in msg.tool_calls]
                updated_chat_history.append(updated_msg)
            elif isinstance(msg, ToolMessage):
                updated_chat_history.append({"type": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
            elif isinstance(msg, SystemMessage):
                updated_chat_history.append({"type": "system", "content": msg.content})


        relevant_docs_for_response = []
        if final_state.get('relevant_docs'):
            for doc in final_state['relevant_docs']:
                if isinstance(doc, Document):                                  # Ensure it's a LangChain Document
                    relevant_docs_for_response.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                else:                                                           # Handle cases where it might already be a dict or other format
                    relevant_docs_for_response.append(doc)

        logger.info(f"Agent responded: '{final_ai_response[:100]}...'")
        return ChatResponse(response=final_ai_response, chat_history=updated_chat_history, clarifying_question=None, relevant_docs=relevant_docs_for_response)            # clarifying_question is None for normal responses

    except HTTPException as e:
        CHAT_ERRORS_TOTAL.labels(error_type="http_exception").inc() # Corrected error counter
        CHAT_LATENCY_HISTOGRAM.observe(time.time() - start_time) # Observe latency even on error
        logger.error(f"HTTP Exception during chat request: {e.detail}", exc_info=True)
        raise
    
    except Exception as e:
        CHAT_ERRORS_TOTAL.labels(error_type="internal_server_error").inc() # Corrected error counter
        CHAT_LATENCY_HISTOGRAM.observe(time.time() - start_time) # Observe latency even on error
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    
    finally:                # Decrement active requests gauge and observe latency histogram in finally block. finally block ensures these operations run whether an exception occurred or not.
        ACTIVE_CHAT_REQUESTS_GAUGE.dec()



@router.post("/feedback")                                       # POST endpoint to receive and store user feedback.
def submit_feedback(request: FeedbackRequest):

    logger.info(f"Received feedback for session '{request.session_id}': {request.feedback_type}")
    try:
        store_feedback(request)                                     # Call the storage function defined in feedback.py
        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store feedback.")
