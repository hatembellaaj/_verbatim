import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional


DEFAULT_PRICING_PATH = Path(__file__).resolve().parent.parent / "openai_pricing.json"


def load_pricing(pricing_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Load OpenAI pricing from a JSON file.

    If the file is missing or invalid, an empty dictionary is returned so the
    UI can still let the user override costs manually.
    """

    path = pricing_path or DEFAULT_PRICING_PATH

    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            logging.warning("Le fichier de pricing n'est pas au format attendu (dict)")
            return {}
    except FileNotFoundError:
        logging.warning("Fichier de pricing %s introuvable", path)
        return {}
    except Exception as exc:  # pragma: no cover - lecture robuste
        logging.warning("Impossible de charger le pricing (%s)", exc)
        return {}


def get_model_cost(model: str, pricing: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    """Return (input_cost, output_cost) for a given model if available."""

    details = pricing.get(model, {}) if pricing else {}
    return details.get("input_per_1k_tokens"), details.get("output_per_1k_tokens")


def format_cost(input_cost: Optional[float], output_cost: Optional[float]) -> str:
    """Format the input/output token costs for display."""

    if input_cost is None and output_cost is None:
        return "Tarif indisponible — mettez à jour via API ou saisie manuelle"

    parts = []
    if input_cost is not None:
        parts.append(f"Entrée : ${input_cost:.4f} / 1k tokens")
    if output_cost is not None:
        parts.append(f"Sortie : ${output_cost:.4f} / 1k tokens")
    return " · ".join(parts)
