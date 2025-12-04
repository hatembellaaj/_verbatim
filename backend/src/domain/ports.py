from abc import ABC, abstractmethod
from typing import Iterable

from .entities import AnalysisResult, Verbatim


class VerbatimRepository(ABC):
    @abstractmethod
    def add(self, verbatim: Verbatim) -> None:
        ...

    @abstractmethod
    def list(self) -> Iterable[Verbatim]:
        ...


class AnalyzerService(ABC):
    @abstractmethod
    def analyze(self, verbatim: Verbatim) -> AnalysisResult:
        ...
