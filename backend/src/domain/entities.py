from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Verbatim:
    id: str
    author: str
    content: str
    created_at: datetime


@dataclass
class AnalysisResult:
    verbatim_id: str
    themes: List[str]
    score: float
    summary: str
