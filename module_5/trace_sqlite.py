"""Instrumented RAG with OpenTelemetry tracing that persists to SQLite."""

import sqlite3

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# --- Set up the tracer provider BEFORE importing starter ---


class SQLiteSpanExporter(SpanExporter):
    """Custom exporter that writes each finished span to a SQLite DB."""

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
            """
        )
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True


provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

from openai import OpenAI  # noqa: E402

from gitsource import GithubRepositoryDataReader  # noqa: E402
from minsearch import Index  # noqa: E402

from rag_helper import RAGBase  # noqa: E402

COMMIT = "8c1834d"

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = OpenAI()

INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


def calc_price(usage):
    input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


class RAGTraced(RAGBase):
    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            result = self.index.search(query, num_results=num_results)
            span.set_attribute("returned", len(result))
            return result

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            input_messages = [
                {"role": "developer", "content": self.instructions},
                {"role": "user", "content": prompt},
            ]
            response = self.llm_client.responses.create(
                model=self.model,
                input=input_messages,
            )

            usage = response.usage
            cost = calc_price(usage)
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("total_tokens", usage.total_tokens)
            span.set_attribute("input_cost", cost["input_cost"])
            span.set_attribute("output_cost", cost["output_cost"])
            span.set_attribute("total_cost", cost["total_cost"])

            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            return response.output_text


rag = RAGTraced(index=index, llm_client=client)


if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)