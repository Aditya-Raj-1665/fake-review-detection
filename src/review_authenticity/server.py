"""Production entry point with an explicit worker count."""

import os

import uvicorn


def main() -> None:
    """Start a bounded ASGI server without development reload or review access logs."""
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if not 1 <= workers <= 16:
        raise ValueError("WEB_CONCURRENCY must be between 1 and 16.")
    uvicorn.run(
        "review_authenticity.api:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        workers=workers,
        access_log=False,
        limit_concurrency=64,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
