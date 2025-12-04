from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from application.use_cases import analyze_verbatim, create_verbatim, list_verbatims
from domain.entities import Verbatim
from infrastructure.repositories.memory import InMemoryVerbatimRepository
from infrastructure.services.dummy_analyzer import DummyAnalyzer

app = FastAPI(title="Verbatim Analyzer API", version="0.1.0")
repo = InMemoryVerbatimRepository()
analyzer = DummyAnalyzer()


class VerbatimIn(BaseModel):
    author: str
    content: str


class VerbatimOut(BaseModel):
    id: str
    author: str
    content: str
    created_at: str

    @classmethod
    def from_entity(cls, entity: Verbatim) -> "VerbatimOut":
        return cls(
            id=entity.id,
            author=entity.author,
            content=entity.content,
            created_at=entity.created_at.isoformat(),
        )


class AnalysisOut(BaseModel):
    verbatim_id: str
    themes: list[str]
    score: float
    summary: str


@app.post("/verbatims", response_model=VerbatimOut)
def add_verbatim(payload: VerbatimIn) -> VerbatimOut:
    created = create_verbatim(repo, author=payload.author, content=payload.content)
    return VerbatimOut.from_entity(created)


@app.get("/verbatims", response_model=list[VerbatimOut])
def get_verbatims() -> list[VerbatimOut]:
    return [VerbatimOut.from_entity(v) for v in list_verbatims(repo)]


@app.post("/verbatims/{verbatim_id}/analysis", response_model=AnalysisOut)
def analyze(verbatim_id: str) -> AnalysisOut:
    verbatim = next((v for v in repo.list() if v.id == verbatim_id), None)
    if not verbatim:
        raise HTTPException(status_code=404, detail="Verbatim introuvable")
    result = analyze_verbatim(analyzer, verbatim)
    return AnalysisOut(**result.__dict__)
