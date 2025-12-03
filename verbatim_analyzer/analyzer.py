import os
import uuid
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import umap
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from verbatim_analyzer.database import save_analysis_result
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

logging.basicConfig(level=logging.INFO)

# Initialisation du client OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
# Remplace par ton propre assistant_id
assistant_id = os.getenv("ASSISTANT_ID")

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

import time


def extract_themes_with_openai(texts_public, texts_private=None, nb_clusters=5):
    if not client:
        raise ValueError("Client OpenAI non initialisé.")

    if texts_private:
        verbatims_concat = [
            f"{pub} {priv}" for pub, priv in zip(texts_public, texts_private)
        ]
    else:
        verbatims_concat = texts_public

    verbatims_concat = verbatims_concat[:50]
    verbatims_concaténés = "\n".join(verbatims_concat)

    prompt = f"""
Tu es un assistant expert en analyse de retours clients.

Je vais te donner une liste d'avis clients (verbatims), chacun contenant un contenu public et privé déjà concaténés.

Ta mission est de :
1. Regrouper les avis selon des thématiques récurrentes.
2. Générer exactement {nb_clusters} *thèmes* principaux (champ `theme`).
3. Pour chaque thème, générer une liste de *sous-thèmes* (champ `subthemes`), sous forme de mots ou phrases courtes.

### Contraintes de sortie :
- Retourne strictement une **liste Python** contenant des **dictionnaires** avec les clés : `theme` et `subthemes`.
- Ne retourne **aucun texte explicatif**, **aucune phrase** en dehors de la liste.
- Structure exacte :
```python
[
  {{ "theme": "Nom du thème 1", "subthemes": ["Sous-thème A", "Sous-thème B"] }},
  {{ "theme": "Nom du thème 2", "subthemes": ["Sous-thème C"] }}
]
```

Voici les verbatims à analyser :

{verbatims_concaténés}
"""

    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=prompt.strip()
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            if run_status.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                if messages.data:
                    try:
                        content = messages.data[0].content[0].text.value.strip()
                        logging.info(
                            "\ud83d\udcc5 Réponse brute de l'assistant (succès) :\n%s",
                            content,
                        )
                    except Exception:
                        raise RuntimeError(
                            "\u2705 Analyse terminée mais la réponse est invalide ou non accessible."
                        )
                else:
                    raise RuntimeError(
                        "\u2705 Analyse terminée mais aucun message reçu de l'assistant."
                    )
                break
            elif run_status.status in ["failed", "cancelled"]:
                last_messages = client.beta.threads.messages.list(thread_id=thread.id)
                error_details = ""
                if last_messages.data:
                    try:
                        error_details = (
                            last_messages.data[0].content[0].text.value.strip()
                        )
                        logging.error(
                            "\ud83d\udcc5 Réponse brute de l'assistant (échec) :\n%s",
                            error_details,
                        )
                    except Exception:
                        error_details = (
                            "Impossible d'extraire le message d'erreur détaillé."
                        )
                        logging.error(
                            "\ud83d\udcc5 Erreur lors de l'extraction du message de l'assistant."
                        )
                raise RuntimeError(
                    f"❌ L'exécution de l'assistant a échoué : {run_status.status}\n\nDétail : {error_details}"
                )
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        content = messages.data[0].content[0].text.value.strip()

        print("\u2705 Thèmes générés par l'assistant :", content)

        # Nettoyage du contenu reçu
        cleaned_content = re.sub(r"```(?:python)?", "", content)
        cleaned_content = cleaned_content.replace("```", "").strip()

        try:
            themes = eval(cleaned_content)  # ⚠️ ou json.loads(cleaned_content) si tu préfères
            if not isinstance(themes, list):
                raise ValueError("La réponse n'est pas une liste valide.")
        except Exception as e:
            raise ValueError(
                f"Erreur de parsing des thèmes : {e}\nContenu reçu : {cleaned_content}"
            )


        return themes

    except Exception as e:
        import traceback

        logging.error(
            "\u274c Erreur lors de l'extraction des thèmes avec OpenAI : %s", str(e)
        )
        logging.error(traceback.format_exc())
        raise


def run_analysis(file, use_openai, user_themes, model_choice, nb_clusters):
    try:
        df = pd.read_csv(file)
        logging.info("\u2705 Dimensions df original : %s", df.shape)

        df.columns = [col.strip() for col in df.columns]

        if "Verbatim public" not in df.columns:
            raise ValueError("La colonne 'Verbatim public' est requise.")
        if "Verbatim privé" not in df.columns:
            df["Verbatim privé"] = ""

        texts = (
            df["Verbatim public"].fillna("") + " " + df["Verbatim privé"].fillna("")
        ).tolist()

        texts_public = df["Verbatim public"].astype(str).tolist()
        texts_private = (
            df["Verbatim privé"].astype(str).tolist()
            if "Verbatim privé" in df.columns
            else None
        )

        if use_openai:
            if not client:
                raise ValueError("Clé API OpenAI manquante.")
            themes = extract_themes_with_openai(
                texts_public, texts_private, nb_clusters
            )
            print(f"\u2705 Thèmes générés : {themes}")
        else:
            themes = [t.strip() for t in user_themes.split(",") if t.strip()]
            print(f"\u2705 Thèmes manuels : {themes}")

        if not themes:
            raise ValueError("Aucun thème n'a été extrait.")

        model_name = (
            "all-MiniLM-L6-v2" if model_choice == "MiniLM" else "all-mpnet-base-v2"
        )
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)

        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)

        df_result = df.copy()
        df_result["Cluster"] = clusters
        df_result["x"] = reduced_embeddings[:, 0]
        df_result["y"] = reduced_embeddings[:, 1]

        # Étape : Enrichir avec les scores de sentiments pour chaque (sub)thème
        df_result = enrich_with_sentiment_scores(df_result, themes)

        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = file.name

        result_path = save_analysis_result(analysis_id, df_result)

        metadata = {
            "id": analysis_id,
            "date": timestamp,
            "model": model_choice,
            "themes": ",".join([t["theme"] for t in themes]),
            "nb_clusters": nb_clusters,
            "filename": filename,
            "result_path": result_path,
        }

        logging.info("\u2705 Dimensions df_result : %s", df_result.shape)
        df_result["Verbatim"] = df_result["Verbatim public"]

        logging.info("\ud83d\udce6 Résultat df_result :")
        logging.info(df_result.head().to_string())

        logging.info("\ud83d\udce6 Metadata :")
        logging.info(metadata)

        return df_result, metadata

    except Exception as e:
        import traceback

        logging.error("\u274c Main program : Erreur pendant l'analyse : %s", str(e))
        logging.error(traceback.format_exc())
        raise

# Initialisation du pipeline de sentiment (modèle multilingue, peut être changé)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def enrich_with_sentiment_scores(df_result, themes):
    """
    Analyse chaque verbatim avec le modèle de sentiment et associe les sous-thèmes
    """
    sentiment_scores = []
    subtheme_columns = []

    for theme in themes:
        for sub in theme['subthemes']:
            col_name = f"{theme['theme']}::{sub}"
            subtheme_columns.append(col_name)

    # ✅ Assure que tous les verbatims sont bien des strings
    df_result["Verbatim"] = df_result["Verbatim"].fillna("").astype(str)

    for text in df_result["Verbatim"]:
        row = {}
        try:
            sentiment = sentiment_pipeline(text[:512])[0]  # tronque à 512 tokens max
        except Exception as e:
            logging.warning(f"⚠️ Erreur de prédiction sur le texte : {text[:50]}... — {e}")
            sentiment = {'label': '3 stars', 'score': 0.5}

        score = {
            '1 star': -1.0,
            '2 stars': -0.5,
            '3 stars': 0.0,
            '4 stars': 0.5,
            '5 stars': 1.0
        }.get(sentiment['label'], 0.0)

        for col in subtheme_columns:
            row[col] = score

        sentiment_scores.append(row)

    sentiment_df = pd.DataFrame(sentiment_scores)
    df_combined = pd.concat([df_result.reset_index(drop=True), sentiment_df], axis=1)
    return df_combined


def generate_heatmap_by_cluster_and_subcluster(df_enriched):
    subtheme_cols = [col for col in df_enriched.columns if '::' in col]

    if not subtheme_cols:
        st.warning("Aucune colonne de sous-thèmes trouvée pour la heatmap.")
        return

    st.write("🎯 Colonnes avec '::' :", subtheme_cols)
    st.dataframe(df_enriched[subtheme_cols].head())

    fig, ax = plt.subplots(figsize=(18, max(8, len(df_enriched) * 0.3)))
    sns.heatmap(df_enriched[subtheme_cols], cmap="RdYlGn", center=0, annot=False, ax=ax)
    ax.set_title("Heatmap des sentiments par sous-thèmes")
    plt.tight_layout()
    st.pyplot(fig)




def generate_heatmap_by_clusters_only(df_enriched):
    subtheme_cols = [col for col in df_enriched.columns if '::' in col]

    if not subtheme_cols:
        st.warning("Aucune colonne de sous-thèmes trouvée pour la heatmap.")
        return

    st.write("🎯 Colonnes avec '::' :", subtheme_cols)
    st.dataframe(df_enriched[subtheme_cols].head())

    cluster_names = list(set(col.split('::')[0] for col in subtheme_cols))

    df_cluster_means = pd.DataFrame()
    for theme in cluster_names:
        cluster_cols = [col for col in subtheme_cols if col.startswith(f"{theme}::")]
        df_cluster_means[theme] = df_enriched[cluster_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_cluster_means) * 0.3)))
    sns.heatmap(df_cluster_means, cmap="RdYlGn", center=0, annot=False, ax=ax)
    ax.set_title("Heatmap des sentiments par thème (agrégé)")
    plt.tight_layout()
    st.pyplot(fig)

def run_marketing_analysis(df, themes, model_choice, nb_clusters):
    try:
        logging.info(f"📄 Fichier reçu pour analyse marketing — {df.shape[0]} lignes")
        df.columns = [col.strip() for col in df.columns]

        if "Verbatim public" not in df.columns or "Note globale avis 1" not in df.columns:
            raise ValueError("Les colonnes 'Verbatim public' et 'Note globale avis 1' sont requises.")

        logging.info(f"📝 Note globale moyenne : {df['Note globale avis 1'].mean():.2f}")
        logging.info(f"🔑 Thèmes reçus : {themes}")
        texts = (df["Verbatim public"].fillna("") + " " + df.get("Verbatim privé", "").fillna("")).tolist()

        model_name = "all-MiniLM-L6-v2" if model_choice == "MiniLM" else "all-mpnet-base-v2"
        logging.info(f"⚙️ Modèle de sentence embedding : {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)

        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)

        df_result = df.copy()
        df_result["Cluster"] = clusters
        df_result["x"] = reduced_embeddings[:, 0]
        df_result["y"] = reduced_embeddings[:, 1]
        df_result["Verbatim"] = df_result["Verbatim public"]

        logging.info("✅ Clustering UMAP + KMeans terminé")

        df_result = enrich_with_sentiment_scores(df_result, themes)
        logging.info(f"✅ Enrichissement sentiment terminé — {df_result.shape[0]} lignes")

        return df_result, {
            "nb_clusters": nb_clusters,
            "model": model_choice,
            "themes": themes
        }

    except Exception as e:
        import traceback
        logging.error("❌ Erreur dans run_marketing_analysis : %s", str(e))
        logging.error(traceback.format_exc())
        raise

