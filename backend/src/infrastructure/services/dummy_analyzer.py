from domain.entities import AnalysisResult, Verbatim
from domain.ports import AnalyzerService


class DummyAnalyzer(AnalyzerService):
    """Exemple d'adaptateur : remplacer par un appel réel à OpenAI/HF."""

    def analyze(self, verbatim: Verbatim) -> AnalysisResult:
        themes = ["satisfaction", "service client"] if "service" in verbatim.content.lower() else ["produit"]
        score = 0.8 if "bien" in verbatim.content.lower() else 0.5
        summary = verbatim.content[:200]
        return AnalysisResult(verbatim_id=verbatim.id, themes=themes, score=score, summary=summary)
