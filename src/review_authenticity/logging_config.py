"""Small structured logger; submitted reviews never enter log records."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Encode operational events and server-side exceptions as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str = "INFO") -> None:
    """Install one JSON handler for application and training events."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(handlers=[handler], level=level, force=True)
