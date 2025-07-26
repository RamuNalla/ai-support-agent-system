from prometheus_client import Counter, Histogram, Gauge

# Define counters for chat requests and errors
CHAT_REQUESTS_TOTAL = Counter(
    "chat_requests_total",
    "Total number of chat requests processed",
    ["status"] # Label: 'status' (e.g., "success", "error", "clarify")
)

CHAT_ERRORS_TOTAL = Counter(
    "chat_errors_total",
    "Total number of errors encountered during chat processing",
    ["error_type"] # Label: 'error_type' (e.g., "http_exception", "internal_server_error", "tool_execution_error")
)

# Define a histogram for chat request latency
CHAT_LATENCY_HISTOGRAM = Histogram(
    "chat_request_duration_seconds",
    "Histogram of chat request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, float('inf')], # Define buckets for latency distribution
)

# Define a gauge for the number of active requests
ACTIVE_CHAT_REQUESTS_GAUGE = Gauge(
    "chat_active_requests",
    "Number of active chat requests being processed"
)

# Histogram for RAG retrieval latency
RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Histogram of RAG document retrieval duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float('inf')] # Example buckets for retrieval
)

# Counter for tool calls
TOOL_CALL_COUNTER = Counter(
    "tool_calls_total",
    "Total number of tool calls",
    ["tool_name", "status"] # Labels: 'tool_name' (e.g., "calculator", "weather"), 'status' (e.g., "success", "error")
)

# --- NEW/MOVED: HTTP Request Metrics Definitions ---
# Counter for total HTTP requests, labeled by method, path, and status code
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total', 'Total number of HTTP requests', ['method', 'path', 'status_code']
)

# Histogram for HTTP request duration in seconds, labeled by method and path
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'Duration of HTTP requests in seconds',
    ['method', 'path'],
    buckets=(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
)
# --- End NEW/MOVED ---
