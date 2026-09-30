"""Errors with actionable messages at application boundaries."""


class DatasetValidationError(ValueError):
    """Signal invalid or insufficient training data."""


class ArtifactError(RuntimeError):
    """Signal a missing, incompatible, or damaged trusted model artifact."""
