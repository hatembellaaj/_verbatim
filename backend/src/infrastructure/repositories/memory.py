from typing import Iterable, List

from domain.entities import Verbatim
from domain.ports import VerbatimRepository


class InMemoryVerbatimRepository(VerbatimRepository):
    def __init__(self) -> None:
        self._storage: List[Verbatim] = []

    def add(self, verbatim: Verbatim) -> None:
        self._storage.append(verbatim)

    def list(self) -> Iterable[Verbatim]:
        return list(self._storage)
