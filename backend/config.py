import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("VOICE_AGENT_MODEL", "minimax-m3:cloud")
    temperature: float = float(os.getenv("VOICE_AGENT_TEMPERATURE", "0.2"))
    max_history: int = int(os.getenv("VOICE_AGENT_MAX_HISTORY", "8"))
    allowed_origins: list[str] = None
    default_session_id: str = os.getenv("VOICE_AGENT_DEFAULT_SESSION", "default")

    def __post_init__(self):
        object.__setattr__(
            self,
            "allowed_origins",
            (self.allowed_origins or os.getenv("ALLOWED_ORIGINS", "*").split(",")),
        )


settings = Settings()
