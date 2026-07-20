"""Module 6 — Instrument pydantic-ai agent with Logfire + Console.

Sends spans to Logfire cloud AND prints them locally so we can count them.
"""

import os
import sys

import logfire

logfire.configure(
    send_to_logfire=True,
    token=os.environ.get("LOGFIRE_TOKEN"),
)
logfire.instrument_pydantic_ai()

# Add ConsoleSpanExporter so we can also see spans locally
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

provider = trace.get_tracer_provider()
if hasattr(provider, "add_span_processor"):
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent import faq_agent, SearchDeps  # noqa: E402
from ingest import build_index, load_faq_data  # noqa: E402


def main():
    documents = load_faq_data()
    index = build_index(documents)
    deps = SearchDeps(index=index)

    questions = [
        "How do I run Ollama locally?",
        "I just discovered the course. Can I join it?",
        "When does the course start?",
    ]

    question = questions[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    print(f"\n=== Question: {question} ===\n", flush=True)

    from pydantic_ai import ModelSettings

    result = faq_agent.run_sync(
        question,
        deps=deps,
        model_settings=ModelSettings(max_tokens=4096),
    )
    print("\n=== Answer ===\n", flush=True)
    print(result.output)


if __name__ == "__main__":
    main()