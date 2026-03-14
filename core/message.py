from dataclasses import dataclass, field
from datetime import datetime, UTC


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
