"""Realistic Logfire traces with full nesting — count tables."""

import dlt


@dlt.resource(name="traces", write_disposition="replace")
def traces():
    """Realistic Logfire trace response — captures nested JSON structures."""
    yield [
        {
            "trace_id": "trace-001",
            "start_timestamp": "2026-07-20T19:00:00Z",
            "end_timestamp": "2026-07-20T19:00:30Z",
            "duration_ms": 30000,
            "spans": [
                {
                    "span_id": "abc123",
                    "trace_id": "trace-001",
                    "name": "invoke_agent faq_agent",
                    "start_timestamp": "2026-07-20T19:00:00Z",
                    "end_timestamp": "2026-07-20T19:00:30Z",
                    "parent_span_id": None,
                    "kind": "internal",
                    "attributes": {
                        "gen_ai.agent.name": "faq_agent",
                        "gen_ai.usage.input_tokens": 7000,
                        "gen_ai.usage.output_tokens": 600,
                    },
                    "otel_events": [
                        {
                            "name": "gen_ai.choice",
                            "timestamp": "2026-07-20T19:00:01Z",
                            "attributes": {
                                "gen_ai.choice.index": 0,
                                "gen_ai.choice.message": {
                                    "role": "assistant",
                                    "content": "Let me search the FAQ first.",
                                    "tool_calls": [
                                        {
                                            "id": "call_001",
                                            "type": "function",
                                            "function": {
                                                "name": "search",
                                                "arguments": '{"query": "ollama locally"}',
                                            },
                                        }
                                    ],
                                },
                            },
                        }
                    ],
                    "otel_links": [],
                    "resource_attributes": {
                        "service.name": "llm-zoomcamp",
                        "telemetry.sdk.language": "python",
                    },
                }
            ],
        }
    ]


@dlt.resource(name="spans", write_disposition="replace")
def spans():
    """Top-level spans resource (alternative endpoint)."""
    yield [
        {
            "span_id": "abc123",
            "trace_id": "trace-001",
            "parent_span_id": None,
            "name": "invoke_agent faq_agent",
            "start_timestamp": "2026-07-20T19:00:00Z",
            "end_timestamp": "2026-07-20T19:00:30Z",
            "kind": "internal",
            "attributes": {
                "gen_ai.agent.name": "faq_agent",
                "gen_ai.usage.input_tokens": 7000,
                "gen_ai.usage.output_tokens": 600,
            },
            "otel_events": [
                {
                    "name": "gen_ai.choice",
                    "timestamp": "2026-07-20T19:00:01Z",
                    "attributes": {
                        "gen_ai.choice.index": 0,
                        "gen_ai.choice.message": {
                            "role": "assistant",
                            "content": "Let me search the FAQ first.",
                            "tool_calls": [
                                {
                                    "id": "call_001",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "ollama locally"}',
                                    },
                                }
                            ],
                        },
                    },
                }
            ],
            "otel_links": [],
            "resource_attributes": {
                "service.name": "llm-zoomcamp",
                "telemetry.sdk.language": "python",
            },
        },
        {
            "span_id": "def456",
            "trace_id": "trace-001",
            "parent_span_id": "abc123",
            "name": "chat gpt-5.4-mini",
            "start_timestamp": "2026-07-20T19:00:01Z",
            "end_timestamp": "2026-07-20T19:00:08Z",
            "kind": "client",
            "attributes": {
                "gen_ai.request.model": "gpt-5.4-mini",
                "gen_ai.usage.input_tokens": 7000,
                "gen_ai.usage.output_tokens": 600,
            },
            "otel_events": [
                {
                    "name": "gen_ai.choice",
                    "timestamp": "2026-07-20T19:00:08Z",
                    "attributes": {
                        "gen_ai.choice.index": 0,
                        "gen_ai.choice.message": {
                            "role": "assistant",
                            "content": "Here is how to run Ollama locally...",
                            "tool_calls": [],
                        },
                    },
                }
            ],
            "otel_links": [],
            "resource_attributes": {
                "service.name": "llm-zoomcamp",
                "telemetry.sdk.language": "python",
            },
        },
    ]


if __name__ == "__main__":
    pipeline = dlt.pipeline(
        pipeline_name="logfire_pipeline_mock",
        destination="duckdb",
        dataset_name="agent_traces",
    )
    load_info = pipeline.run([spans(), traces()])
    print(load_info)

    print("\nAll tables in agent_traces schema:")
    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'agent_traces' ORDER BY table_name"
        ) as cur:
            tables = [row[0] for row in cur.fetchall()]
            for t in tables:
                print(f"  - {t}")
            print(f"\nTotal: {len(tables)}")