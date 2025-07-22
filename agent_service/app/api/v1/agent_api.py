import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.core import Agent, AgentState
from app.config.settings import settings

logger = logging.getLogger(__name__)                # Initialize logger

router = APIRouter()                                # Create an API router for agent-related endpoints (helps organizing endpoints)

compiled_agent_graph = None                     # Initialize to None
try:
    agent_instance = Agent(gemini_api_key=settings.GEMINI_API_KEY)
    compiled_agent_graph = agent_instance.build_graph()
    logger.info("LangGraph agent and graph initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangGraph agent: {e}", exc_info=True)

class ChatRequest(BaseModel):               # Pydantic model for incoming chat requests.
    query: str

class ChatResponse(BaseModel):              # Pydantic model for outgoing chat responses.
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):        # asyn function that handles the chat request. Receives a user query and returns the agent's response.

    if not compiled_agent_graph:
        logger.error("Agent graph not initialized. Cannot process chat request.")
        raise HTTPException(status_code=503, detail="AI agent service is not ready.")

    logger.info(f"Received chat query: '{request.query}'")

    try:
        initial_state = AgentState(messages=[HumanMessage(content=request.query)])          # Initial state for the graph
        final_state = await compiled_agent_graph.ainvoke(initial_state)                     # Lanngraph agent is invoked asyncronously
        ai_response_message = final_state['messages'][-1]                                   # Extract the last message, which should be the AI's response
        response_content = ai_response_message.content

        logger.info(f"Agent responded: '{response_content[:100]}...'")
        return ChatResponse(response=response_content)

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

