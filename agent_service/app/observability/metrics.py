from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNTER = Counter(                      # Define counters for chat requests and errors
    "chat_requests_total",                      # Metric name
    "Total number of chat requests processed",  # Description
    ["status"]                                  # Label: 'status' (e.g., "success", "failure")
)

ERROR_COUNTER = Counter(
    "chat_errors_total",
    "Total number of errors encountered during chat processing",
    ["error_type"]                              # Label: 'error_type' (e.g., "http_exception", "internal_server_error")
)

CHAT_LATENCY_HISTOGRAM = Histogram(                 # Define a histogram for chat request latency
    "chat_request_duration_seconds",
    "Histogram of chat request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],  # Define buckets for latency distribution
)

ACTIVE_REQUESTS_GAUGE = Gauge(                      # Define a gauge for the number of active requests
    "chat_active_requests",
    "Number of active chat requests being processed"
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Histogram of RAG document retrieval duration"
)
TOOL_CALL_COUNTER = Counter(
    "tool_calls_total",
    "Total number of tool calls",
    ["tool_name"]
)
