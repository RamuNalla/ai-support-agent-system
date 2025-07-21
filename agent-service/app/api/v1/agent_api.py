import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.core import Agent, AgentState
from app.config.settings import settings

logger = logging.getLogger(__name__)                # Initialize logger

router = APIRouter()                                # Create an API router for agent-related endpoints (helps organizing endpoints)

# Initialize the LangGraph agent globally
# This ensures the LLM and graph are set up once when the app starts
try:
    agent_instance = Agent(api_key=settings.GEMINI_API_KEY)
    compiled_agent_graph = agent_instance.build_graph()
    logger.info("LangGraph agent and graph initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangGraph agent: {e}")
    compiled_agent_graph = None             # Ensure it's None if initialization fails

class ChatRequest(BaseModel):               # Pydantic model for incoming chat requests.
    query: str

class ChatResponse(BaseModel):              # Pydantic model for outgoing chat responses.
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Endpoint to chat with the AI agent.
    Receives a user query and returns the agent's response.
    """
    if not compiled_agent_graph:
        logger.error("Agent graph not initialized. Cannot process chat request.")
        raise HTTPException(status_code=503, detail="AI agent service is not ready.")

    logger.info(f"Received chat query: '{request.query}'")

    try:
        # Initial state for the graph
        initial_state = AgentState(messages=[HumanMessage(content=request.query)])

        # Invoke the compiled graph
        # The .invoke() method runs the graph from its entry point
        # and returns the final state.
        final_state = await compiled_agent_graph.ainvoke(initial_state)

        # Extract the last message, which should be the AI's response
        ai_response_message = final_state['messages'][-1]
        response_content = ai_response_message.content

        logger.info(f"Agent responded: '{response_content[:100]}...'")
        return ChatResponse(response=response_content)

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

