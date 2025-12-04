from datetime import datetime
from typing import Iterable
from uuid import uuid4

from domain.entities import AnalysisResult, Verbatim
from domain.ports import AnalyzerService, VerbatimRepository


def create_verbatim(repo: VerbatimRepository, author: str, content: str) -> Verbatim:
    verbatim = Verbatim(id=str(uuid4()), author=author, content=content, created_at=datetime.utcnow())
    repo.add(verbatim)
    return verbatim


def list_verbatims(repo: VerbatimRepository) -> Iterable[Verbatim]:
    return repo.list()


def analyze_verbatim(analyzer: AnalyzerService, verbatim: Verbatim) -> AnalysisResult:
    return analyzer.analyze(verbatim)
