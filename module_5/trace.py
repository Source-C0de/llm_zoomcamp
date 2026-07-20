"""Instrumented RAG with OpenTelemetry tracing.

Wraps rag(), search(), and llm() each in their own span using
the ConsoleSpanExporter so we can see what OTel captures.
Also records token usage and cost as span attributes.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# --- Set up the tracer provider BEFORE importing starter ---
provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

from openai import OpenAI  # noqa: E402

from gitsource import GithubRepositoryDataReader  # noqa: E402
from minsearch import Index  # noqa: E402

from rag_helper import RAGBase  # noqa: E402

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
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


# Pricing for openai/gpt-oss-120b (per 1M tokens)
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


def calc_price(usage):
    """Compute cost from token usage."""
    input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


class RAGTraced(RAGBase):
    """Subclass of RAGBase that wraps key methods in their own spans."""

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

            # --- Token usage + cost as span attributes ---
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
