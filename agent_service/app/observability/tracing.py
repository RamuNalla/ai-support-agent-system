import os
from opentelemetry import trace, metrics
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

    trace_processor = BatchSpanProcessor(span_exporter)
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(trace_processor)
    trace.set_tracer_provider(trace_provider)
    
    # --- Metrics Setup (NEW) ---
    # metric_exporter = OTLPMetricExporter( # Exporter for metrics
    #     endpoint=otlp_endpoint,
    #     insecure=True
    # )
    # metric_reader = PeriodicExportingMetricReader( # Reader to periodically export metrics
    #     exporter=metric_exporter,
    #     export_interval_millis=5000 # Export every 5 seconds
    # )
    # meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    # metrics.set_meter_provider(meter_provider) # Set the global MeterProvider

    # --- Instrumentation ---
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace_provider) # Pass trace_provider
    LangchainInstrumentor().instrument(tracer_provider=trace_provider) # Pass trace_provider
    RequestsInstrumentor().instrument()
    
    logger.info(f"OpenTelemetry tracing and metrics set up. Exporter endpoint: {otlp_endpoint}")