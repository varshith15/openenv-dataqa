"""
Root-level server entry point for OpenEnv compatibility.
"""

from dataqa_env.server.app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
