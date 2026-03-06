import logging
import os
import re
import sqlite3
from typing import Tuple

import pandas as pd
from openai import OpenAI


logger = logging.getLogger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None


PROMPT_TEMPLATE = """
Tu es un expert SQL.
Convertis la demande utilisateur en UNE requête SQL SQLite valide.

Contraintes strictes:
- Table unique: enriched_reviews
- Ne retourner QUE la requête SQL (sans markdown, sans explication)
- Utiliser uniquement les colonnes fournies ci-dessous
- Si une colonne contient un espace, un accent ou des caractères spéciaux (ex: ::), l'entourer de guillemets doubles
- Requête en lecture seule (SELECT)
- Limiter les résultats à 200 lignes maximum sauf si COUNT/GROUP BY

Colonnes disponibles:
{columns}

Demande utilisateur:
{question}
""".strip()


FORBIDDEN_SQL_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\battach\b",
    r"\bpragma\b",
]


def _auto_quote_column_identifiers(sql_query: str, available_columns: list[str]) -> str:
    """Ajoute des guillemets doubles autour des noms de colonnes si nécessaire.

    Cela permet de supporter les colonnes contenant espaces, accents ou caractères spéciaux
    (ex: "Note IA", "Programmation::Programmation variée et de qualité").
    """
    patched = sql_query
    for col in sorted(available_columns, key=len, reverse=True):
        if not col:
            continue
        escaped = re.escape(col)
        patched = re.sub(
            rf'(?<!")({escaped})(?!")',
            lambda m: f'"{m.group(1)}"',
            patched,
        )
    return patched


def _extract_sql(response_text: str) -> str:
    """Nettoie la réponse LLM pour isoler la requête SQL."""
    if not response_text:
        raise ValueError("Réponse vide du modèle.")

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL)
    sql = fenced.group(1).strip() if fenced else response_text.strip()

    sql = re.sub(r"^sql\s+", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";").strip() + ";"

    lowered = sql.lower().lstrip()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise ValueError("La requête générée n'est pas une requête SELECT/WITH autorisée.")

    body_without_last_semicolon = sql[:-1]
    if ";" in body_without_last_semicolon:
        raise ValueError("Plusieurs requêtes détectées. Une seule requête SQL est autorisée.")

    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError("La requête contient une opération SQL interdite.")

    return sql


def generate_sql_from_question(question: str, available_columns: list[str], model: str) -> str:
    if not client:
        raise ValueError("OPENAI_API_KEY manquante : impossible d'interroger OpenAI.")

    if not question or not question.strip():
        raise ValueError("Question vide.")

    prompt = PROMPT_TEMPLATE.format(columns=", ".join(available_columns), question=question.strip())
    logger.info("[chat-sql] Génération SQL demandée | model=%s | nb_colonnes=%s", model, len(available_columns))

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        timeout=45,
        messages=[
            {"role": "system", "content": "Tu transformes des questions métier en SQL SQLite sûr."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content if response.choices else ""
    sql = _extract_sql(content)
    logger.info("[chat-sql] Requête SQL générée avec succès")
    return sql


def run_sql_on_dataframe(df: pd.DataFrame, sql_query: str) -> Tuple[pd.DataFrame, str]:
    logger.info("[chat-sql] Exécution SQL sur dataframe enrichi | rows=%s", len(df))
    patched_sql = _auto_quote_column_identifiers(sql_query, df.columns.tolist())
    if patched_sql != sql_query:
        logger.info("[chat-sql] Requête SQL adaptée automatiquement pour les noms de colonnes spéciaux")

    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("enriched_reviews", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(patched_sql, conn)
        logger.info("[chat-sql] Exécution SQL terminée | rows_result=%s", len(result_df))
        return result_df, patched_sql
    finally:
        conn.close()
