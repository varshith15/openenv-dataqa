"""
FastAPI application for the DataQA Environment.

Usage:
    uvicorn dataqa_env.server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
    from .environment import DataQAEnvironment
    from ..models import DataQAAction, DataQAObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from dataqa_env.server.environment import DataQAEnvironment
    from dataqa_env.models import DataQAAction, DataQAObservation

app = create_app(
    DataQAEnvironment, DataQAAction, DataQAObservation, env_name="dataqa_env"
)


@app.get("/")
def root():
    """Root endpoint — environment info."""
    return {
        "name": "DataQA Environment",
        "description": "Two-phase data quality assurance environment: identify issues + propose fixes",
        "tasks": ["easy", "medium", "hard", "alignment", "coding", "toolcalling", "moderation"],
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
