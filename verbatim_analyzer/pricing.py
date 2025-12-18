import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import streamlit as st


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


def estimate_average_chars(verbatims: list[str]) -> int:
    """Estimate the average number of characters in provided verbatims."""

    if not verbatims:
        return 0

    cleaned = [v or "" for v in verbatims]
    total_chars = sum(len(v) for v in cleaned)
    return int(total_chars / len(cleaned)) if cleaned else 0


def estimate_sampling_tokens(sample_size: int, avg_chars_per_verbatim: int) -> tuple[float, float]:
    """Estimate tokens per verbatim and total input tokens for a sampled batch."""

    if sample_size <= 0 or avg_chars_per_verbatim <= 0:
        return 0.0, 0.0

    tokens_per_verbatim = max(avg_chars_per_verbatim / 4, 1)
    total_tokens = tokens_per_verbatim * sample_size
    return tokens_per_verbatim, total_tokens


def estimate_input_cost(total_tokens: float, input_cost_per_1k: Optional[float]) -> float:
    """Estimate the input cost given total tokens and a price per 1k tokens."""

    if not input_cost_per_1k or total_tokens <= 0:
        return 0.0

    return (total_tokens / 1000) * input_cost_per_1k


def render_llm_selector(label_prefix: str = "OpenAI") -> tuple[str, float, float]:
    """Render LLM + pricing pickers in the main area (not only sidebar).

    Returns the tuple (model, input_cost, output_cost) using pricing defaults
    when available and falling back to stored session choices.
    """

    pricing = load_pricing()
    model_choices = sorted(pricing.keys()) or ["gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]

    default_model = st.session_state.get("llm_model") or ("gpt-4o-mini" if "gpt-4o-mini" in model_choices else model_choices[0])
    model = st.selectbox(
        f"Modèle {label_prefix}",
        model_choices,
        index=model_choices.index(default_model) if default_model in model_choices else 0,
        key=f"{label_prefix}_llm_model_main",
    )

    default_input, default_output = get_model_cost(model, pricing)
    col_a, col_b = st.columns(2)
    with col_a:
        input_cost = st.number_input(
            "Coût entrée / 1k tokens ($)",
            min_value=0.0,
            value=float(default_input or st.session_state.get("llm_pricing", {}).get("input", 0.0)),
            format="%.6f",
            key=f"{label_prefix}_llm_input_main",
        )
    with col_b:
        output_cost = st.number_input(
            "Coût sortie / 1k tokens ($)",
            min_value=0.0,
            value=float(default_output or st.session_state.get("llm_pricing", {}).get("output", 0.0)),
            format="%.6f",
            key=f"{label_prefix}_llm_output_main",
        )

    st.caption(format_cost(input_cost, output_cost))

    st.session_state["llm_model"] = model
    st.session_state["llm_pricing"] = {
        "input": input_cost,
        "output": output_cost,
        "source": "manual-main",
    }

    return model, input_cost, output_cost
