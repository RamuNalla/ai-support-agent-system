import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging

logger = logging.getLogger(__name__)

def setup_tracing(app):                 # Sets up OpenTelemetry tracing for the FastAPI app. It configures the TracerProvider, a span processor, and an exporter. It also instruments FastAPI, LangChain, and requests libraries.
    
    logger.info("Setting up OpenTelemetry tracing...")

    service_name = os.getenv("OTEL_SERVICE_NAME", "agent-service")      # Get the service name from environment variables, defaulting to 'agent-service'
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")

    resource = Resource.create({                        # Resource defines attributes about the service emitting telemetry
        SERVICE_NAME: service_name
    })

    span_exporter = OTLPSpanExporter(                   # Create an OTLP Span Exporter, This sends traces to the OpenTelemetry Collector
        endpoint=otlp_endpoint,
        insecure=True                                   # Use insecure for local development 
    )

    processor = BatchSpanProcessor(span_exporter)       # Configure the TracerProvider with a BatchSpanProcessor. It collects spans and sends them in batches for efficiency
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)                 # Set the TracerProvider as the global provider. This makes the tracer available throughout your application

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)       # Instrument FastAPI to automatically create spans for incoming requests
    LangchainInstrumentor().instrument(tracer_provider=provider)              # Instrument LangChain to trace LangGraph nodes, LLM calls, and tool execution
    RequestsInstrumentor().instrument()                                     # Instrument the requests library for any outgoing HTTP calls
    
    logger.info(f"OpenTelemetry tracing set up. Exporter endpoint: {otlp_endpoint}")

