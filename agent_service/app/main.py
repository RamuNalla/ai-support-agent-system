import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, REGISTRY
from app.api.v1 import agent_api
from app.observability.logging_config import setup_logging
from app.observability.tracing import setup_tracing 
from app.config.settings import settings
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, start_http_server
import time
from app.observability.metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS, ACTIVE_CHAT_REQUESTS_GAUGE # Import metrics

setup_logging()                                 # Set up logging for the application
logger = logging.getLogger(__name__)

PROMETHEUS_METRICS_PORT = 8001


@asynccontextmanager                            # lifespan context manager for startup and shutdown events
async def lifespan(app: FastAPI):
    logger.info("AI Support Agent Service is starting up...")
    
    try:
        start_http_server(PROMETHEUS_METRICS_PORT) # NEW: Start the Prometheus metrics server
        logger.info(f"Prometheus metrics server started on port {PROMETHEUS_METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}", exc_info=True)
        # Continue starting the app, but metrics won't be exposed

    setup_tracing(app) # Call the tracing setup
    
    yield # Application will run until this point
    
    logger.info("AI Support Agent Service is shutting down.")

app = FastAPI(                                  # Initialize FastAPI application, passing the lifespan context manager
    title="AI Support Agent Service",
    description="A production-grade AI support agent with LangGraph and FastAPI.",
    version="1.0.0",
    lifespan=lifespan,                          # Use the lifespan context manager
)

if settings.TRACING_ENABLED: # Assuming you add this to settings.py
    setup_tracing(app)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to capture HTTP request metrics (total count and duration)
    and track active requests using a Gauge.
    This runs for every incoming HTTP request to the FastAPI application.
    """
    # Increment active requests when a request starts
    ACTIVE_CHAT_REQUESTS_GAUGE.inc() 
    start_time = time.time() # Record the start time of the request
    
    response = await call_next(request) # Process the request and get the response
    
    process_time = time.time() - start_time # Calculate the duration
    method = request.method
    path = request.url.path
    status_code = response.status_code

    # Increment the HTTP request counter with relevant labels
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=status_code).inc()
    
    # Observe the HTTP request duration in the histogram
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(process_time)
    
    # Decrement active requests when a request finishes
    ACTIVE_CHAT_REQUESTS_GAUGE.dec()
    
    return response # Return the response to the client


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(REGISTRY))

app.include_router(agent_api.router, prefix="/api/v1", tags=["Agent"])          # Mounts the agent_api router under /api/v1 prefix (makes the /chat endpoint accessible at /api/v1/chat)

@app.get("/health", summary="Health Check", response_model=dict)                # Health checkpoint (useful for Kubernetes readiness/liveness probes)
async def health_check():                                                       # Returns a simple health status
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "message": "AI Support Agent Service is running."}

