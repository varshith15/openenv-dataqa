"""Entrypoint for openenv-core deployment. Delegates to dataqa_env.server.app."""

from dataqa_env.server.app import app  # noqa: F401


def main():
    """Start the environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
