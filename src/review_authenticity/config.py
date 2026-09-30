"""Validated training and runtime settings."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseModel):
    """Record the complete reproducible classical training configuration."""

    seed: int = 42
    validation_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    max_features: int = Field(default=30000, ge=10)
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = Field(default=1, ge=1)
    max_df: float = Field(default=1.0, gt=0, le=1)
    logistic_c: float = Field(default=1.0, gt=0)
    class_weight: Literal["balanced"] | None = "balanced"
    nb_alpha: float = Field(default=1.0, gt=0)
    calibration_folds: int = Field(default=3, ge=2, le=10)
    selection_metric: Literal["f1", "precision", "recall", "accuracy", "macro_f1"] = "f1"
    threshold: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def check_ratios(self) -> "TrainingConfig":
        """Reject splits and n-grams that cannot produce a useful training set."""
        if self.validation_ratio + self.test_ratio >= 0.8:
            raise ValueError("Validation and test ratios must leave more than 20% for training.")
        if not 1 <= self.ngram_range[0] <= self.ngram_range[1] <= 5:
            raise ValueError("ngram_range must satisfy 1 <= minimum <= maximum <= 5.")
        return self


class Settings(BaseSettings):
    """Read API configuration once, at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)
    app_env: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model_type: Literal["baseline", "distilbert"] = "baseline"
    model_path: Path = Path("artifacts/baseline/demo-v1")
    prediction_threshold: float | None = Field(default=None, gt=0, lt=1)
    max_review_length: int = Field(default=10000, ge=10, le=100000)
    max_batch_size: int = Field(default=32, ge=1, le=128)
    max_request_bytes: int = Field(default=1048576, ge=1024, le=16777216)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @model_validator(mode="after")
    def check_cors(self) -> "Settings":
        """Keep production CORS restricted to explicit origins."""
        if self.app_env == "production" and "*" in self.cors_origins:
            raise ValueError("Production CORS_ORIGINS must list explicit origins.")
        return self
