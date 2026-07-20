"""dlt pipeline that pulls data from Logfire into DuckDB."""

import os

import dlt
from dlt.sources.rest_api import RESTAPIConfig, rest_api_resources

LOGFIRE_READ_TOKEN = os.environ.get("LOGFIRE_READ_TOKEN")
LOGFIRE_PROJECT_URL = os.environ.get(
    "LOGFIRE_PROJECT_URL", "https://logfire-us.pydantic.dev/v1/"
)


@dlt.source
def logfire_source(
    access_token: str = dlt.secrets.value,
    project_url: str = LOGFIRE_PROJECT_URL,
):
    config: RESTAPIConfig = {
        "client": {
            "base_url": project_url,
            "auth": {"type": "bearer", "token": access_token},
        },
        "resources": [
            {
                "name": "traces",
                "endpoint": {
                    "path": "traces/search",
                    "method": "POST",
                    "json": {"limit": 100},
                    "data_selector": "traces",
                },
            },
            {
                "name": "spans",
                "endpoint": {
                    "path": "spans/search",
                    "method": "POST",
                    "json": {"limit": 100},
                    "data_selector": "spans",
                },
            },
        ],
    }
    yield from rest_api_resources(config)


if __name__ == "__main__":
    pipeline = dlt.pipeline(
        pipeline_name="logfire_pipeline",
        destination="duckdb",
        dataset_name="agent_traces",
    )
    load_info = pipeline.run(logfire_source(access_token=LOGFIRE_READ_TOKEN))
    print(load_info)