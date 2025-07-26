import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Dict, Any, List, Optional
from app.agent.core import Agent, AgentState
from app.config.settings import settings
from app.observability.feedback import store_feedback
from app.observability.metrics import (             # Import defined metrics
    REQUEST_COUNTER, 
    ERROR_COUNTER, 
    CHAT_LATENCY_HISTOGRAM, 
    ACTIVE_REQUESTS_GAUGE,
    RAG_RETRIEVAL_LATENCY,
    TOOL_CALL_COUNTER
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

class FeedbackRequest(BaseModel): 
    session_id: str
    message_content: str
    feedback_type: str              # "positive" or "negative"
    comment: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, agent: Agent = Depends(get_agent)):
    logger.info(f"Received chat request: {request.message}")
    ACTIVE_REQUESTS_GAUGE.inc()             # Increment when request starts
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
            raise HTTPException(status_code=500, detail="Agent returned an empty response (no final state).")
        
        # Check if 'messages' key exists and is not empty
        if 'messages' not in final_state:
            logger.error(f"Final state from agent graph is missing 'messages' key. Final state type: {type(final_state)}, Content: {final_state}")
            raise HTTPException(status_code=500, detail="Agent returned an invalid state (missing messages key).")

        if not final_state['messages']:
            logger.error(f"Final state 'messages' list is empty. Final state type: {type(final_state)}, Content: {final_state}")
            raise HTTPException(status_code=500, detail="Agent returned an empty messages list.")

        clarifying_q = final_state.get('clarifying_question')        # Check for clarifying question first 
        if clarifying_q:
            logger.info(f"Agent asked a clarifying question: '{clarifying_q}'")
            return ChatResponse(response="", chat_history=[], clarifying_question=clarifying_q)     # Return the clarifying question directly to the user


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

        REQUEST_COUNTER.labels(status="success").inc()
        logger.info(f"Agent responded: '{final_ai_response[:100]}...'")
        return ChatResponse(response=final_ai_response, chat_history=updated_chat_history, clarifying_question=None)            # clarifying_question is None for normal responses

    except HTTPException as e:
        ERROR_COUNTER.labels(error_type=e.detail).inc()                     # For a caught HTTP exception, increment the error counter with specific type
        raise e
    
    except Exception as e:
        ERROR_COUNTER.labels(error_type="internal_server_error").inc()      # For an unexpected exception, increment the error counter with a generic type
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    
    finally:                # Decrement active requests gauge and observe latency histogram in finally block. finally block ensures these operations run whether an exception occurred or not.
        
        ACTIVE_REQUESTS_GAUGE.dec()                             # Decrement when request finishes
        end_time = time.time()                                  # Record end time
        CHAT_LATENCY_HISTOGRAM.observe(end_time - start_time)   # Observe the duration



@router.post("/feedback")                                       # POST endpoint to receive and store user feedback.
def submit_feedback(request: FeedbackRequest):

    logger.info(f"Received feedback for session '{request.session_id}': {request.feedback_type}")
    try:
        store_feedback(request)                                 # Call the storage function defined in feedback.py
        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store feedback.")
