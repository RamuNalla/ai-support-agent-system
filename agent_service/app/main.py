import logging
from fastapi import FastAPI
from app.api.v1 import agent_api
from app.observability.logging_config import setup_logging
from contextlib import asynccontextmanager

setup_logging()                                 # Set up logging for the application
logger = logging.getLogger(__name__)

@asynccontextmanager                            # lifespan context manager for startup and shutdown events
async def lifespan(app: FastAPI):
    logger.info("AI Support Agent Service is starting up...")
    # Any startup code can go here
    yield
    # Any shutdown code can go here
    logger.info("AI Support Agent Service is shutting down.")

app = FastAPI(                                  # Initialize FastAPI application, passing the lifespan context manager
    title="AI Support Agent Service",
    description="A production-grade AI support agent with LangGraph and FastAPI.",
    version="1.0.0",
    lifespan=lifespan,                          # Use the lifespan context manager
)

app.include_router(agent_api.router, prefix="/api/v1", tags=["Agent"])          # Mounts the agent_api router under /api/v1 prefix (makes the /chat endpoint accessible at /api/v1/chat)

@app.get("/health", summary="Health Check", response_model=dict)                # Health checkpoint (useful for Kubernetes readiness/liveness probes)
async def health_check():                                                       # Returns a simple health status
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "message": "AI Support Agent Service is running."}

