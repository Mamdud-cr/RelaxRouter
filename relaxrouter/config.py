"""Environment-based local configuration."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data: Path = Path(os.getenv("RELAXROUTER_DATA", "data"))
    device: str = os.getenv("RELAXROUTER_DEVICE", "auto")
    carbon_intensity: float = float(os.getenv("RELAXROUTER_CARBON_INTENSITY", "0.16"))
    auto_continue: bool = os.getenv("RELAXROUTER_AUTO_CONTINUE", "0") == "1"
    retry_limit: int = int(os.getenv("RELAXROUTER_RETRY_LIMIT", "1"))
    confidence_threshold: float = float(os.getenv("RELAXROUTER_CONFIDENCE_THRESHOLD", "0.05"))
    max_batch_size: int = int(os.getenv("RELAXROUTER_MAX_BATCH_SIZE", "8"))
