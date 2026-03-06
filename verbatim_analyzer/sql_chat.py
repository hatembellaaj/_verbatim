import os
import re
import sqlite3
from typing import Tuple

import pandas as pd
from openai import OpenAI


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None


PROMPT_TEMPLATE = """
Tu es un expert SQL.
Convertis la demande utilisateur en UNE requête SQL SQLite valide.

Contraintes strictes:
- Table unique: enriched_reviews
- Ne retourner QUE la requête SQL (sans markdown, sans explication)
- Utiliser uniquement les colonnes fournies ci-dessous
- Requête en lecture seule (SELECT)
- Limiter les résultats à 200 lignes maximum sauf si COUNT/GROUP BY

Colonnes disponibles:
{columns}

Demande utilisateur:
{question}
""".strip()


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

    return sql


def generate_sql_from_question(question: str, available_columns: list[str], model: str) -> str:
    if not client:
        raise ValueError("OPENAI_API_KEY manquante : impossible d'interroger OpenAI.")

    if not question or not question.strip():
        raise ValueError("Question vide.")

    prompt = PROMPT_TEMPLATE.format(columns=", ".join(available_columns), question=question.strip())

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Tu transformes des questions métier en SQL SQLite sûr."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content if response.choices else ""
    return _extract_sql(content)


def run_sql_on_dataframe(df: pd.DataFrame, sql_query: str) -> Tuple[pd.DataFrame, str]:
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("enriched_reviews", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(sql_query, conn)
        return result_df, sql_query
    finally:
        conn.close()
