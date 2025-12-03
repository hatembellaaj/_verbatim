import time
import logging
import pandas as pd
from openai import OpenAI
import os
from fpdf import FPDF
import tempfile

# Récupération des clés d'API
openai_api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None




def exporter_rapport_pdf(texte: str, titre: str = "Rapport Synthèse Verbatims") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_title(titre)
    pdf.multi_cell(0, 10, texte)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file.read()


def generer_rapport_openai(stats_df: pd.DataFrame) -> str:
    """
    Génère un rapport de synthèse marketing à partir d'un tableau analysé.
    """
    if not client:
        logging.error("❌ OpenAI client non initialisé.")
        return ""

    try:
        df_reset = stats_df.reset_index()

        if "Sous-thème" not in df_reset.columns and df_reset.columns.size > 0:
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "Sous-thème"})

        preferred_cols = ["Sous-thème", "mean", "Polarité estimée", "Véracité polarité"]
        selected_cols = [col for col in preferred_cols if col in df_reset.columns]

        if len(selected_cols) < 2:
            remaining = [c for c in df_reset.columns if c not in ["Sous-thème"]]
            selected_cols = ["Sous-thème"] + remaining[:3]

        resume = df_reset[selected_cols].to_string(index=False)

        prompt = f"""
Tu es un expert en marketing et en analyse de l'expérience client.

Voici un tableau synthétique des sous-thèmes identifiés à partir de verbatims clients.
Chaque ligne indique un sous-thème, la moyenne des notes associées, la polarité estimée par IA, et une étiquette de véracité.

Ta tâche :
1. Résume les grands enseignements de cette analyse et donne les chiffres qui justifie le classement déjà fait et les priorités annoncés.
2. Mets en avant les points positifs et négatifs les plus marquants.
3. Propose des recommandations pratiques et actionnables.

Voici les données :
{resume}

Donne une synthèse claire, concise, en français.
"""

        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt.strip())
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled"]:
                raise RuntimeError(f"OpenAI a échoué : {run_status.status}")
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        texte = messages.data[0].content[0].text.value.strip()
        return texte

    except Exception as e:
        logging.exception("❌ Erreur dans la génération du rapport")
        return ""
