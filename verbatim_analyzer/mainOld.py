import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import re
import platform
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from verbatim_analyzer.database import init_db
from verbatim_analyzer.marketing_analyzer import (
    extract_marketing_clusters_with_openai,
    associer_sous_themes_par_similarity
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from verbatim_analyzer.report_generator import generer_rapport_openai, exporter_rapport_pdf
from typing import List, Optional
import datetime as dt
import matplotlib.patches as patches
import plotly.express as px
from openai import OpenAI
client = OpenAI()

# Chargement du modèle RoBERTa (multilingue)
analyser_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Liste étendue de mots/expressions négatives (ajustable)
REGEX_MOTS_NEGATIFS = re.compile(
    r"\b(manque|nul|horrible|catastrophique|décevant|trop long|attente interminable|mal organisé|inutile|déplorable|pas\s+(terrible|bon|correct|agréable)|rien de spécial|très cher|arnaque)\b",
    flags=re.IGNORECASE
)


def construire_matrice_finale(df_enriched, incoherences, subtheme_cols, note_col="Note globale avis 1"):
    matrice = pd.DataFrame(index=df_enriched.index)

    for col in subtheme_cols:
        for idx in df_enriched.index:
            row = incoherences[(incoherences["Verbatim"] == df_enriched.at[idx, "Verbatim complet"]) &
                               (incoherences["Sous-thème"] == col)]
            if not row.empty and row.iloc[0]["Decision"] == "Non retenue":
                matrice.at[idx, col] = np.nan
            else:
                matrice.at[idx, col] = df_enriched.at[idx, note_col] if pd.notna(df_enriched.at[idx, col]) else np.nan

    matrice.insert(0, "Verbatim complet", df_enriched["Verbatim complet"])
    matrice.insert(1, "Note globale", df_enriched[note_col])

    # ✅ Log de suivi
    logging.info(f"📐 Matrice finale construite : {matrice.shape[0]} lignes × {matrice.shape[1]} colonnes")

    return matrice

# Mots-clés négatifs ou positifs à surveiller (ajustable)
MOTS_NEGATIFS = ["manque", "insuffisant", "mauvais", "problème", "attente", "cher", "décevant"]
MOTS_POSITIFS = ["magique", "féérique", "excellent", "bravo", "super", "parfait"]

def coherence_note_mot(note, sous_theme):
    """Retourne False si la note est incohérente avec le sens du sous-thème."""
    st_lower = sous_theme.lower()

    if note >= 5 and any(m in st_lower for m in MOTS_NEGATIFS):
        return False
    if note <= 2 and any(m in st_lower for m in MOTS_POSITIFS):
        return False
    return True





def verifier_coherence_semantique(
    df,
    subtheme_cols,
    seuil=0.3,
    alpha=0.9,              # poids MiniLM (0.9 = 90% MiniLM, 10% TF-IDF)
    model_name="all-MiniLM-L6-v2"
):
    """
    Vérifie la cohérence verbatim ↔ sous-thème via MiniLM + TF-IDF + logique note/mots.
    Retourne un DataFrame long avec Verbatim, Sous-thème, Similarité (pondérée), Cohérence.
    - seuil : si score final < seuil → "⚠️ Suspect"
    - alpha : pondération MiniLM (0.9 MiniLM + 0.1 TF-IDF par défaut)
    - élimination directe si note contradictoire avec polarité implicite du sous-thème
    """

    # 🔹 Listes de mots pour détecter le ton des sous-thèmes
    MOTS_NEGATIFS = ["manque", "insuffisant", "mauvais", "problème", "attente", "cher", "décevant", "raté"]
    MOTS_POSITIFS = ["magique", "féérique", "excellent", "bravo", "super", "parfait"]

    def coherence_note_mot(note, sous_theme: str) -> bool:
        """Retourne False si la note est incohérente avec le sous-thème."""
        st_lower = sous_theme.lower()
        if note >= 5 and any(m in st_lower for m in MOTS_NEGATIFS):
            return False
        if note <= 2 and any(m in st_lower for m in MOTS_POSITIFS):
            return False
        return True

    # --- Collecte verbatims ↔ sous-thèmes ---
    rows = []
    for col in subtheme_cols:
        mask = df[col].notna()
        for idx in df[mask].index:
            rows.append({
                "Verbatim": str(df.at[idx, "Verbatim complet"]),
                "Sous-thème": col,
                "Note globale": df.at[idx, "Note globale avis 1"]
            })

    assignments = pd.DataFrame(rows)
    if assignments.empty:
        return pd.DataFrame()

    verbatims = assignments["Verbatim"].tolist()
    subthemes = assignments["Sous-thème"].tolist()

    # --- TF-IDF ---
    if USE_TFIDF:
        texts = verbatims + subthemes
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(texts)
        verbatim_vecs = X[:len(verbatims)]
        subtheme_vecs = X[len(verbatims):]
        sims_tfidf = [cosine_similarity(verbatim_vecs[i], subtheme_vecs[i])[0][0]
                      for i in range(len(verbatims))]
    else:
        sims_tfidf = [0.0] * len(verbatims)

    # --- MiniLM ---
    model = SentenceTransformer(model_name)
    emb_verbatims = model.encode(verbatims, convert_to_tensor=True)
    emb_subthemes = model.encode(subthemes, convert_to_tensor=True)
    sims_minilm = util.cos_sim(emb_verbatims, emb_subthemes).diagonal().cpu().numpy().tolist()

    # --- Fusion pondérée ---
    sims_final = [alpha * s_minilm + (1 - alpha) * s_tfidf
                  for s_minilm, s_tfidf in zip(sims_minilm, sims_tfidf)]

    # --- Ajout des scores ---
    assignments["Similarité_TFIDF"] = sims_tfidf
    assignments["Similarité_MiniLM"] = sims_minilm
    assignments["Score_final"] = sims_final

    # --- Cohérence finale ---
    decisions = []
    for i, row in assignments.iterrows():
        note = row["Note globale"]
        sous_theme = row["Sous-thème"]
        s_tfidf = row["Similarité_TFIDF"]
        score = row["Score_final"]

        # 1️⃣ Règle lexicale stricte TF-IDF
        if s_tfidf < 0.03:
            decisions.append("Non retenue")
            continue

        # 2️⃣ Règle "note ↔ mot du sous-thème"
        if not coherence_note_mot(note, sous_theme):
            decisions.append("Non retenue")
            continue

        # 3️⃣ Règle de similarité
        if score < seuil:
            decisions.append("⚠️ Suspect")
        else:
            decisions.append("OK")

    assignments["Cohérence"] = decisions
    assignments["Decision"] = decisions  # alias pour compatibilité

    return assignments



def tracer_bulles_corr_accel_plotly(stats: pd.DataFrame, top_labels: int = 15, inclure_extremes: bool = True):
    if stats.empty:
        st.warning("Aucune donnée pour tracer le graphe.")
        return

    data = stats.reset_index().rename(columns={"Sous-thème":"Sous-theme"})
    data["Catégorie"] = np.where(data["is_recent_first"], "Premières occurrences récentes", "Occurrences établies")

    # Sélection des labels (moins de bruit)
    to_label = set(data.nlargest(top_labels, "prop")["Sous-theme"])
    if inclure_extremes:
        to_label |= set(data.nsmallest(5, "corr")["Sous-theme"])
        to_label |= set(data.nlargest(5, "corr")["Sous-theme"])
        to_label |= set(data.nsmallest(5, "accel")["Sous-theme"])
        to_label |= set(data.nlargest(5, "accel")["Sous-theme"])
    data["label"] = np.where(data["Sous-theme"].isin(to_label), data["Sous-theme"], "")

    fig = px.scatter(
        data,
        x="corr", y="accel",
        size="prop", size_max=48,
        color="Catégorie",
        color_discrete_map={
            "Premières occurrences récentes": "#ffb74d",
            "Occurrences établies": "#f2f2f2",
        },
        hover_name="Sous-theme",
        hover_data={
            "corr":":.2f", "accel":":.2f",
            "prop":":.1%", "first_seen": True,
            "Sous-theme": False  # déjà dans hover_name
        }
    )

    # Quadrillage / zones colorées comme ta maquette
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.add_shape(type="rect", x0=-1, x1=0, y0=-1e9, y1=1e9, fillcolor="#f8d7da", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=-1e9, y1=1e9, fillcolor="#d4edda", opacity=0.25, line_width=0)

    # Petites étiquettes uniquement pour la sélection "to_label"
    #for _, r in data[data["label"]!=""].iterrows():
    #    fig.add_annotation(x=r["corr"], y=r["accel"], text=r["label"],
    #                       showarrow=False, xanchor="left", yanchor="bottom", xshift=6, yshift=6, font={"size":10})

    fig.update_layout(
        title="Corrélation × Accélération des sous‑thèmes (taille = % d’occurrences)",
        xaxis_title="Corrélation avec la note (− … +)",
        yaxis_title="Accélération des occurrences (pente normalisée)",
        legend_title="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    # bornes X lisibles
    fig.update_xaxes(range=[-1, 1], zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

def construire_matrice_verbatims(df_enriched, subtheme_cols, note_col="Note globale avis 1"):
    """
    Construit une matrice verbatim × sous-thèmes :
    - Lignes = verbatims
    - Colonnes = sous-thèmes
    - Valeur = note du verbatim si rattaché, NaN sinon
    """
    matrice = pd.DataFrame(index=df_enriched.index)

    for col in subtheme_cols:
        # Si le verbatim est rattaché (col non NaN), on met la note
        matrice[col] = np.where(df_enriched[col].notna(), df_enriched[note_col], np.nan)

    # On peut rajouter les colonnes de contexte utiles
    matrice.insert(0, "Verbatim complet", df_enriched["Verbatim complet"])
    matrice.insert(1, "Note globale", df_enriched[note_col])

    return matrice

def _occ_time_series(df, col_date, present_mask, freq="W"):
    ts = (
        df.loc[present_mask, [col_date]]
        .dropna()
        .assign(n=1)
        .set_index(col_date)
        .resample(freq)["n"].sum()
        .fillna(0.0)
    )
    return ts

def verifier_polarite_openai(sous_theme: str, polarite_estimee: str, note_moyenne: float) -> str:
    """
    Vérifie ou corrige la polarité avec OpenAI.
    Retourne "Positive", "Negative" ou "Neutre".
    """
    prompt = f"""
    Voici un sous-thème client : "{sous_theme}"
    Polarité détectée automatiquement : {polarite_estimee}
    Note moyenne associée : {note_moyenne}

    Corrige la polarité en tenant compte du sens réel du sous-thème.
    Réponds uniquement par un mot : Positive, Negative ou Neutre.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        correction = response.choices[0].message.content.strip()
        if correction.lower().startswith("pos"):
            return "Positive"
        elif correction.lower().startswith("neg"):
            return "Negative"
        else:
            return "Neutre"
    except Exception as e:
        logging.warning(f"⚠️ OpenAI indisponible ({e}) → on garde {polarite_estimee}")
        return polarite_estimee


def calculer_bulles_corr_accel(
    df_enriched: pd.DataFrame,
    subtheme_cols: List[str],
    note_col: str = "Note globale avis 1",
    date_col: Optional[str] = "Date avis",
    freq: str = "W",                   # "W", "M", "D"
    recent_k_periods: int = 3          # alerte orange si 1ère occurrence dans les K dernières périodes
) -> pd.DataFrame:
    """
    Retourne un DF indexé par Sous-thème avec:
      - corr : corrélation (présence vs note)
      - accel : pente normalisée des occurrences (accélération)
      - prop : part des occurrences (taille bulle)
      - first_seen : 1ère date d'occurrence
      - is_recent_first : bool alerte "premières occurrences récentes"
    """
    df = df_enriched.copy()
    y = pd.to_numeric(df[note_col], errors="coerce")

    # Taille = % d'occurrences (tous sous-thèmes confondus)
    total_occ = df[subtheme_cols].notna().sum().sum()
    prop = {c: (df[c].notna().sum() / total_occ) if total_occ else 0.0 for c in subtheme_cols}

    # Corrélation point-bisérielle approx (corr Pearson entre 0/1 et note)
    corr = {}
    for c in subtheme_cols:
        x = df[c].notna().astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 3 and x[m].std() > 0 and y[m].std() > 0:
            corr[c] = float(np.corrcoef(x[m], y[m])[0, 1])
        else:
            corr[c] = 0.0

    # Accélération & première occurrence
    accel, first_seen, recent_first = {}, {}, {}
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # borne haute pour savoir si "récent"
        now = (df[date_col].max() or pd.Timestamp.utcnow())
        # taille période pour remonter K périodes
        if freq == "W":
            recent_threshold = now - pd.Timedelta(weeks=recent_k_periods)
        elif freq == "M":
            recent_threshold = now - pd.DateOffset(months=recent_k_periods)
        else:
            recent_threshold = now - pd.Timedelta(days=recent_k_periods)

        for c in subtheme_cols:
            mask = df[c].notna()
            dmin = df.loc[mask, date_col].min()
            first_seen[c] = dmin
            recent_first[c] = bool(pd.notna(dmin) and dmin >= recent_threshold)

            ts = _occ_time_series(df, date_col, mask, freq=freq)
            if len(ts) < 3:
                accel[c] = 0.0
                continue
            x_idx = np.arange(len(ts))
            slope = np.polyfit(x_idx, ts.values, 1)[0]
            std = ts.std()
            accel[c] = float(slope / (std if std > 0 else 1.0))
    else:
        for c in subtheme_cols:
            accel[c] = 0.0
            first_seen[c] = pd.NaT
            recent_first[c] = False

    out = pd.DataFrame({
        "Sous-thème": subtheme_cols,
        "corr": [corr[c] for c in subtheme_cols],
        "accel": [accel[c] for c in subtheme_cols],
        "prop": [prop[c] for c in subtheme_cols],
        "first_seen": [first_seen[c] for c in subtheme_cols],
        "is_recent_first": [recent_first[c] for c in subtheme_cols],
    }).set_index("Sous-thème")

    return out.sort_values("prop", ascending=False)

def tracer_bulles_corr_accel(
    stats: pd.DataFrame,
    title: str = "Corrélation × Accélération des sous‑thèmes",
    top_labels: int = 20
):
    if stats.empty:
        st.warning("Aucune donnée pour tracer le graphe.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Fond en bandes (rouge à gauche, vert à droite) façon maquette
    ax.add_patch(patches.Rectangle((-1.0, ax.get_ylim()[0]), 1.0, 9999, color="#f8d7da", alpha=0.35))
    ax.add_patch(patches.Rectangle((0.0, ax.get_ylim()[0]), 1.0, 9999, color="#d4edda", alpha=0.35))

    # axes 0 (corrélation=0, accélération=0)
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)

    # Taille des bulles
    sizes = (stats["prop"] * 7000).clip(lower=60)

    # Couleur: orange si "récent", sinon gris avec bord violet
    face_colors = np.where(stats["is_recent_first"], "#ffb74d", "#f2f2f2")
    edge_colors = "#7C2A90"

    sc = ax.scatter(
        stats["corr"], stats["accel"], s=sizes,
        c=face_colors, edgecolors=edge_colors, linewidths=2, alpha=0.95
    )

    # Labels pour top bulles
    for name in stats.head(top_labels).index:
        ax.annotate(name, (stats.loc[name, "corr"], stats.loc[name, "accel"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlim(-1.0, 1.0)
    # y auto; mais garde un minimum lisible
    ymin, ymax = stats["accel"].min(), stats["accel"].max()
    pad = max((ymax - ymin) * 0.15, 0.3)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Corrélation avec la note (− … +)")
    ax.set_ylabel("Accélération des occurrences (pente normalisée)")
    ax.set_title(title)

    # Légendes custom
    leg1 = patches.Patch(facecolor="#ffb74d", edgecolor=edge_colors, label="Premières occurrences récentes")
    leg2 = patches.Patch(facecolor="#f2f2f2", edgecolor=edge_colors, label="Occurrences établies")
    ax.legend(handles=[leg1, leg2], loc="upper left", frameon=False)

    st.pyplot(fig)



def contient_mots_negatifs(texte):
    return any(mot in texte.lower() for mot in mots_negatifs)

def analyser_sentiment_mixte(texte):
    try:
        texte = texte.strip()
        if not texte:
            return {"label": "Neutral", "score": 0.0, "source": "vide"}

        # Analyse IA RoBERTa
        res = analyser_sentiment(texte[:512])[0]
        score_label = res['label']
        score = int(score_label.split()[0])

        # Heuristique de mots négatifs
        negatif_detecte = contient_mots_negatifs(texte)

        # Heuristique d'ajustement
        if score >= 4 and negatif_detecte:
            # Avis globalement positif mais éléments critiques → neutre
            return {"label": "Mixed (positif mais critique)", "score": score, "source": "mixte"}
        elif score <= 2 and not negatif_detecte:
            # Score bas mais aucune critique explicite → douteux
            return {"label": "Mixed (note faible mais pas critique)", "score": score, "source": "mixte"}
        elif negatif_detecte and score >= 3:
            return {"label": "Negative", "score": score, "source": "regex négatif"}
        else:
            return {"label": "Positive" if score >= 4 else "Negative" if score <= 2 else "Neutral", "score": score, "source": "RoBERTa"}

    except Exception as e:
        return {"label": "Erreur", "score": 0, "source": str(e)}


logging.basicConfig(level=logging.INFO)

ASSISTANT_MODEL_NAME = "gpt-4-turbo"

st.set_page_config(layout="wide")
st.title("🧠 Analyse des verbatims client")

init_db()

menu = st.sidebar.selectbox("Navigation", ["Marketing"])


# 🔍 Liste de mots ou expressions indiquant un ressenti potentiellement négatif
mots_negatifs = [
    "souhait", "longue", "absence", "manque", "trop long", "dommage", "pas assez", "trop cher", "impossible",
    "file d’attente", "interminable", "aucune", "décevant", "catastrophique",
    "rien pour", "manquait", "ne fonctionne pas", "sans intérêt", "n'a pas aimé",
    "déception", "pas bon", "pas top", "nul", "ennuyeux", "raté", "erreur", "à éviter"
]
USE_TFIDF = True   # ou False si on veut désactiver
TFIDF_ALPHA = 0.3  # poids de TF-IDF dans la pondération


def evaluer_veracite_polarite(row, seuil_bas=3, seuil_haut=4):
    polarite = row["Polarité estimée"]
    note = row["mean"]

    if polarite == "Positive" and note <= seuil_bas:
        return "❌ Faux positif"
    elif polarite == "Negative" and note >= seuil_haut:
        return "❌ Faux négatif"
    elif polarite == "Positive" and note >= seuil_haut:
        return "✅ Vrai positif"
    elif polarite == "Negative" and note <= seuil_bas:
        return "✅ Vrai négatif"
    else:
        return "🟡 Ambigu / mixte"



def afficher_dataframe_propre(df: pd.DataFrame, cmap: str = "Blues"):
    """
    Affiche une dataframe stylisée en supprimant les lignes/colonnes entièrement vides
    et indique combien ont été ignorées. Réinitialise l'index si nécessaire pour éviter les erreurs de style.
    """
    if df.empty:
        st.warning("📭 La table est vide, rien à afficher.")
        return

    # Sauvegarde des tailles avant nettoyage
    lignes_avant = df.shape[0]
    colonnes_avant = df.shape[1]

    # Nettoyage : suppression des lignes/colonnes complètement vides
    df_cleaned = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    lignes_apres = df_cleaned.shape[0]
    colonnes_apres = df_cleaned.shape[1]

    lignes_supprimees = lignes_avant - lignes_apres
    colonnes_supprimees = colonnes_avant - colonnes_apres

    if lignes_supprimees > 0 or colonnes_supprimees > 0:
        st.info(f"🔍 {lignes_supprimees} ligne(s) et {colonnes_supprimees} colonne(s) vides ont été ignorées.")

    if df_cleaned.empty:
        st.warning("🚫 Tous les éléments ont été filtrés (table vide après nettoyage).")
        return

    # ✅ Sécurité : réinitialise index si non unique (nécessaire pour .style)
    if not df_cleaned.index.is_unique:
        df_cleaned = df_cleaned.reset_index()

    if not df_cleaned.columns.is_unique:
        df_cleaned.columns = [f"{col}_{i}" if list(df_cleaned.columns).count(col) > 1 else col
                              for i, col in enumerate(df_cleaned.columns)]

    st.dataframe(df_cleaned.style.background_gradient(cmap=cmap))



def analyser_par_batches(df, batch_size=1000, process_func=None):
    if process_func is None:
        raise ValueError("Vous devez fournir une fonction process_func(batch_df)")

    total = len(df)
    results = []
    for i in range(0, total, batch_size):
        st.info(f"🧪 Traitement du batch {i//batch_size + 1} / {total//batch_size + 1}")
        batch_df = df.iloc[i:i+batch_size].copy()
        try:
            result = process_func(batch_df)
            if not result.empty:
                results.append(result)
        except Exception as e:
            st.error(f"❌ Erreur dans le batch {i//batch_size + 1} : {e}")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Sous-thème", "Polarité estimée", "Véracité polarité"])


def traiter_batch_polarite(batch_df):
    subtheme_cols = [col for col in batch_df.columns if "::" in col]

    if not subtheme_cols:
        return pd.DataFrame(columns=["Sous-thème", "Polarité estimée", "Véracité polarité"])

    # Préparation données fondues
    df_melted_batch = batch_df[["Note globale avis 1"] + subtheme_cols].melt(
        id_vars="Note globale avis 1",
        var_name="Sous-thème",
        value_name="Score"
    ).dropna()

    return enrichir_polarite_veracite(df_melted_batch, batch_df, subtheme_cols)




def enrichir_polarite_veracite(df_melted, df_enriched, subtheme_cols):
    polarite_info = []
    total = len(subtheme_cols)
    progress_bar = st.progress(0, text="📊 Analyse de polarité des sous-thèmes...")

    for i, subtheme in enumerate(subtheme_cols):
        textes = df_enriched[df_enriched[subtheme].notna()]["Verbatim complet"].astype(str)
        if textes.empty:
            continue

        with ThreadPoolExecutor(max_workers=8) as executor:
            result_dicts = list(executor.map(analyser_sentiment_mixte, textes))

        labels = [res["label"] if isinstance(res, dict) and "label" in res else "Neutre" for res in result_dicts]
        moyenne = df_melted[df_melted["Sous-thème"] == subtheme]["Score"].mean()

        try:
            sentiment_label = pd.Series(labels).mode()[0]
        except IndexError:
            sentiment_label = "Neutre"

        polarite = (
            "Positive" if "pos" in sentiment_label.lower()
            else "Negative" if "neg" in sentiment_label.lower()
            else "Neutre"
        )

        # Correction heuristique locale
        if contient_mots_negatifs(subtheme) and polarite == "Positive":
            polarite = "Neutre"

        # ✅ Rectification finale via OpenAI
        polarite = verifier_polarite_openai(subtheme, polarite, moyenne)


        veracite = evaluer_veracite_polarite({
            "Polarité estimée": polarite,
            "mean": moyenne
        })

        polarite_info.append({
            "Sous-thème": subtheme,
            "Polarité estimée": polarite,
            "Véracité polarité": veracite
        })

        progress_bar.progress((i + 1) / total, text=f"📊 Polarité : {subtheme}")

    progress_bar.empty()
    return pd.DataFrame(polarite_info)




def preparer_csv_export(df, filename, label="⬇️ Télécharger CSV"):
    try:
        csv_data = df.to_csv(index=False).encode("utf-8")
        size_kb = len(csv_data) / 1024
        logging.info(f"🗂️ Export CSV — {filename} — taille : {size_kb:.1f} KB")
        st.download_button(label=label, data=csv_data, file_name=filename, mime="text/csv")
    except Exception as e:
        logging.exception("❌ Erreur pendant l'export CSV")
        st.error(f"Erreur export : {e}")


    

if menu == "Marketing":

    st.info("📍 Étape 1 : Début pipeline Marketing")


    mem = psutil.virtual_memory()
    st.code(f"""
    📊 SYSTEM INFO :
    Python: {platform.python_version()}
    OS: {platform.system()}
    RAM utilisée: {mem.used // (1024 ** 2)}MB / {mem.total // (1024 ** 2)}MB
    """)



    st.header("\U0001F4CA Analyse marketing des sous-clusters et notes globales")

    uploaded_file = st.file_uploader("\U0001F4C1 Téléverser un fichier CSV avec notes globales", type="csv")

    # === OPTIONS AFFICHAGE (Marketing) ===
    st.sidebar.markdown("### ⚙️ Options d'affichage")

    is_big_file = uploaded_file and uploaded_file.size > 4_000_000

    show_note_comparison = st.sidebar.checkbox("Afficher comparaison des verbatims utilisés", value=not is_big_file)
    show_subtheme_table = st.sidebar.checkbox("Afficher tableau sous-thèmes filtrés", value=True)
    show_best_worst = st.sidebar.checkbox("Afficher meilleurs/pires sous-thèmes", value=not is_big_file)
    show_distribution_chart = st.sidebar.checkbox("Afficher histogramme des scores", value=not is_big_file)
    show_verbatims_examples = st.sidebar.checkbox("Afficher verbatims représentatifs", value=False)
    show_profil_matrix = st.sidebar.checkbox("Afficher matrice Profil × Sous-thèmes", value=not is_big_file)
    show_matrice_verbatims = st.sidebar.checkbox("Afficher matrice Verbatims × Sous-clusters", value=False)
    show_unassigned = st.sidebar.checkbox("Afficher les verbatims non associés", value=False)
    # Option sidebar pour afficher les incohérences
    show_incoherences = st.sidebar.checkbox("Afficher incohérences sémantiques", value=False)

    use_openai = st.sidebar.checkbox("Utiliser OpenAI pour les clusters")
    nb_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=1000, value=5)
    model_choice = st.sidebar.radio("Modèle d'encodage", ["MiniLM", "BERT"])
    # Choix du seuil de similarité (avec valeur par défaut 0.45)
    seuil_similarite = st.sidebar.slider(
        "Seuil de similarité (MiniLM/BERT)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05
    )

    user_themes = ""
    themes = []

    if not use_openai:
        user_themes = st.sidebar.text_area("Liste manuelle des clusters (format JSON accepté ou simple CSV)")
        
        if not user_themes.strip():
            st.warning("⚠️ Saisissez des thèmes manuels ou activez l'option OpenAI.")
            st.stop()
        
        try:
            # Tentative de chargement en JSON
            parsed = json.loads(user_themes)
            
            if isinstance(parsed, list) and all(isinstance(t, dict) and "theme" in t and "subthemes" in t for t in parsed):
                themes = parsed
                st.success(f"✅ {len(themes)} thème(s) JSON correctement interprété(s)")
                with st.expander("📚 Aperçu des thèmes interprétés"):
                    for theme in themes:
                        st.markdown(f"### 🟦 {theme['theme']}")
                        for sub in theme['subthemes']:
                            st.markdown(f"- {sub}")

            else:
                raise ValueError("❌ Format JSON invalide : chaque élément doit avoir 'theme' et 'subthemes'")
        
        except Exception as e:
            # Si JSON échoue, fallback simple CSV
            themes = [{"theme": t.strip(), "subthemes": []} for t in user_themes.split(",") if t.strip()]
            st.info("ℹ️ Format simple détecté (pas JSON valide) : seuls les noms de thèmes seront utilisés")

    if uploaded_file is not None:
        try:
            st.info("📍 Étape 2 : Chargement du CSV")
            df = pd.read_csv(uploaded_file)
            nb_lignes = df.shape[0]
            st.success(f"✅ Fichier chargé : {nb_lignes} lignes")

            # ➕ Estimation de durée
            estimation_minutes = round(nb_lignes * 0.008)  # ≈ 0.5 sec/ligne
            if estimation_minutes > 1:
                st.warning(f"⏳ Le traitement peut prendre environ **{estimation_minutes} minute(s)** selon la charge.")

        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            st.stop()

        required_cols = ["Verbatim public", "Note globale avis 1"]
        if not all(c in df.columns for c in required_cols):
            st.error("❌ Le fichier doit contenir au moins 'Verbatim public' et 'Note globale avis 1'.")
            st.stop()

        texts_public = df["Verbatim public"].astype(str).fillna("").tolist()
        texts_private = df["Verbatim privé"].astype(str).fillna("").tolist() if "Verbatim privé" in df.columns else [""] * len(texts_public)

        if use_openai:
            with st.spinner("\U0001F52E Extraction des clusters avec OpenAI..."):
                try:
                    themes = extract_marketing_clusters_with_openai(texts_public, texts_private, nb_clusters)
                    st.success("✅ Clusters extraits avec succès")
                    st.markdown("### \U0001F9E0 Thèmes extraits")
                    for t in themes:
                        st.markdown(f"**🟦 {t['theme']}**")
                        for s in t['subthemes']:
                            if isinstance(s, dict):
                                st.markdown(f"- {s['label']}  _(mots-clés: {', '.join(s.get('keywords', []))})_")
                            else:
                                st.markdown(f"- {s}")  # fallback pour compat rétro

                except Exception as e:
                    st.error(f"Erreur OpenAI : {e}")
                    st.stop()


        df_enriched = df.copy()
        df_enriched["Verbatim complet"] = df["Verbatim public"].fillna("") + " " + df.get("Verbatim privé", "").fillna("")

        try:
            model_name = "all-MiniLM-L6-v2" if model_choice == "MiniLM" else "bert-base-nli-mean-tokens"
            logging.info(f"🧠 Modèle sélectionné : {model_name}")

            logging.info(f"🧪 Validation themes — Nombre total : {len(themes)}")
            for t in themes:
                logging.info(f"📂 Thème : {t['theme']} — {len(t['subthemes'])} sous-thèmes")

            assert isinstance(themes, list)
            assert all("subthemes" in t for t in themes)
            assert any(t["subthemes"] for t in themes), "❌ Aucun sous-thème trouvé dans 'themes'"

            with st.spinner("🔄 Analyse marketing en cours, cela peut prendre plusieurs minutes..."):
                df_enriched = associer_sous_themes_par_similarity(
                    df_enriched,
                    themes=themes,
                    text_col="Verbatim complet",
                    model_name=model_name,
                    seuil_similarite=seuil_similarite
                )
            logging.info("✅ Attribution des sous-thèmes réussie")
        except Exception as e:
            logging.exception("❌ Erreur lors de l'attribution des sous-thèmes")
            st.error(f"Erreur lors de l'attribution des sous-thèmes : {e}")
            st.stop()

        subtheme_cols = [col for col in df_enriched.columns if "::" in col]
        if not subtheme_cols:
            st.error("❌ Aucune colonne de sous-thème trouvée.")
            st.stop()

        # 📐 Matrice
        if show_matrice_verbatims:
            matrice = construire_matrice_verbatims(df_enriched, subtheme_cols)
            st.markdown("### 📐 Matrice Verbatims × Clusters / Sous-clusters")
            st.dataframe(matrice)
            preparer_csv_export(matrice, "matrice_verbatims_clusters.csv", "⬇️ Télécharger la matrice complète")

        # 🔍 Vérification de cohérence
        if show_incoherences:
            incoherences = verifier_coherence_semantique(
                df_enriched, subtheme_cols, seuil=0.3, alpha=0.7
            )

            if "Decision" not in incoherences.columns:
                logging.warning("⚠️ Pas de colonne Decision dans incoherences")
                incoherences["Decision"] = "OK"

            st.markdown("### 🔍 Vérification de cohérence verbatim ↔ sous-thème")
            st.dataframe(incoherences)

            # ✅ Export intermédiaire
            preparer_csv_export(incoherences, "incoherences.csv", "⬇️ Télécharger incohérences détectées")



            # Matrice enrichie avec décision finale
            matrice_finale = construire_matrice_finale(df_enriched, incoherences, subtheme_cols)
            matrice_finale["Décision finale"] = np.where(
                matrice_finale[subtheme_cols].notna().any(axis=1),
                "Retenue",
                "Non retenue"
            )

            # Compte clair : uniquement les vrais non retenus
            nb_non_retenus = (matrice_finale["Décision finale"] == "Non retenue").sum()
            nb_retenus = (matrice_finale["Décision finale"] == "Retenue").sum()

            st.info(f"📊 Résultat : {nb_non_retenus} verbatim(s) non retenus — {nb_retenus} retenus")

            st.markdown("### 📐 Matrice Verbatims × Sous-thèmes (avec décision finale)")
            st.dataframe(matrice_finale)

            preparer_csv_export(matrice_finale, "matrice_verbatims_finale.csv",
                                "⬇️ Télécharger matrice finale")

            # 🧹 Suppression des incohérents
            suspects = incoherences[incoherences["Cohérence"] == "⚠️ Suspect"]["Verbatim"].unique()
            # 🧹 Suppression des incohérents et non retenus
            avant = len(df_enriched)
            for _, row in incoherences.iterrows():
                verb = row["Verbatim"]
                sous_theme = row["Sous-thème"]
                if row["Decision"] in ["⚠️ Suspect", "Non retenue"]:
                    df_enriched.loc[df_enriched["Verbatim complet"] == verb, sous_theme] = np.nan

            # Supprimer seulement les verbatims sans aucune association restante
            df_enriched = df_enriched[df_enriched[subtheme_cols].notna().any(axis=1)].copy()
            apres = len(df_enriched)

            #st.info(f"🧹 {avant - apres} verbatim(s) entièrement non associés supprimés. Restant : {apres}")

        # 📊 Stats de base
        df_melted = df_enriched[["Note globale avis 1"] + subtheme_cols].melt(
            id_vars="Note globale avis 1",
            var_name="Sous-thème",
            value_name="Score"
        ).dropna()

        subtheme_stats = df_melted.groupby("Sous-thème")["Score"].agg(["count", "mean"]).sort_values("mean", ascending=False)

        # 🧠 Polarité + véracité
        df_polarite = analyser_par_batches(df_enriched, batch_size=1000, process_func=traiter_batch_polarite)

        if df_polarite.empty or "Sous-thème" not in df_polarite.columns:
            st.error("❌ La polarité n’a pas pu être calculée (table vide).")
            st.stop()

        # 🔗 Fusion
        merged_stats = subtheme_stats.merge(df_polarite, left_index=True, right_on="Sous-thème")
        merged_stats = merged_stats.set_index("Sous-thème")

        # Nombre total de verbatims par note
        # Étape 2 – Comparaison entre total, utilisés (brut) et verbatims uniques exploités
        note_distribution = df["Note globale avis 1"].value_counts().sort_index()
        used_notes = df_melted["Note globale avis 1"].value_counts().sort_index()

        # Verbatims uniques qui ont été associés à au moins un sous-thème
        used_unique_notes = (
            df_enriched.loc[df_enriched[subtheme_cols].notna().any(axis=1), "Note globale avis 1"]
            .value_counts()
            .sort_index()
        )

        note_comparison = pd.DataFrame({
            "Total verbatims": note_distribution,
            "Utilisés dans l’analyse (associations)": used_notes,
            "Verbatims uniques utilisés": used_unique_notes
        }).fillna(0).astype(int)

        if show_note_comparison:
            st.markdown("### 🧾 Comparaison : verbatims disponibles vs. utilisés dans l’analyse")
            st.dataframe(note_comparison.style.background_gradient(cmap="Blues"))

            # Vérification des verbatims non associés à un sous-thème
            nb_unassigned = df_enriched[subtheme_cols].isna().all(axis=1).sum()
            verbatims_non_assignes = df_enriched[df_enriched[subtheme_cols].isna().all(axis=1)]
            st.warning(f"⚠️ {nb_unassigned} verbatim(s) n'ont été associés à aucun sous-thème.")

            if show_unassigned:
                st.markdown("### 📭 Verbatims non associés à un cluster/sous-cluster")
                st.write(f"Nombre total : {len(verbatims_non_assignes)}")
                st.dataframe(verbatims_non_assignes[["Verbatim public", "Note globale avis 1"]])
                
                # ✅ Option d’export CSV
                preparer_csv_export(
                    verbatims_non_assignes[["Verbatim public", "Note globale avis 1"]],
                    "verbatims_non_assignes.csv",
                    label="⬇️ Télécharger les verbatims non associés"
                )


        st.write("🧾 Aperçu des données analysées (df_melted)", df_melted.head(20))
        st.write("📊 Distribution des scores :", df_melted["Score"].value_counts())

        # Moyenne et fréquence par sous-thème
        subtheme_stats = df_melted.groupby("Sous-thème")["Score"].agg(["count", "mean"]).sort_values("mean", ascending=False)

        # Polarité & véracité (IA)
        df_polarite = analyser_par_batches(df_enriched, batch_size=1000, process_func=traiter_batch_polarite)


        # Fusion des infos
        merged_stats = subtheme_stats.merge(df_polarite, left_index=True, right_on="Sous-thème")
        merged_stats = merged_stats.set_index("Sous-thème")
        merged_stats["count"] = pd.to_numeric(merged_stats["count"], errors="coerce")
        merged_stats["mean"] = pd.to_numeric(merged_stats["mean"], errors="coerce")
        merged_stats = merged_stats.dropna(subset=["mean", "count"])

        # 🔍 Aperçu complet avant filtrage
        st.write("📌 Aperçu avant filtrage sur la véracité", merged_stats[["count", "mean", "Polarité estimée", "Véracité polarité"]])
        st.write("🔍 Distribution des véracités :", merged_stats["Véracité polarité"].value_counts())

        # ✅ Extraire les vrais positifs/négatifs
        filtered_stats = merged_stats[merged_stats["Véracité polarité"].isin(["✅ Vrai positif", "✅ Vrai négatif"])]

        if filtered_stats.empty:
            st.warning("⚠️ Aucun sous-thème avec une polarité jugée 'vrai positif' ou 'vrai négatif'. Vérifiez l'algorithme ou les seuils.")
            st.stop()

        nb_ambigus = len(merged_stats) - len(filtered_stats)
        st.info(f"{nb_ambigus} sous-thème(s) ambigus ont été exclus de l’analyse des plus polarisants.")

        max_mean = filtered_stats["mean"].max()
        min_mean = filtered_stats["mean"].min()

        best_subs = filtered_stats[filtered_stats["mean"] == max_mean]
        worst_subs = filtered_stats[filtered_stats["mean"] == min_mean]
        if show_best_worst:
            st.markdown("### 🏅 Sous-thèmes les plus polarisants (avec polarité IA)")

            # 💚 Favorables
            st.success(f"💚 Moyenne MAX = {max_mean:.2f}")
            for name, row in best_subs.iterrows():
                st.markdown(f"- **{name}** — {int(row['count'])} verbatims")
                st.markdown(f"  ↪️ Polarité : `{row['Polarité estimée']}` — `{row['Véracité polarité']}`")

            # 💔 Défavorables
            st.error(f"💔 Moyenne MIN = {min_mean:.2f}")
            for name, row in worst_subs.iterrows():
                st.markdown(f"- **{name}** — {int(row['count'])} verbatims")
                st.markdown(f"  ↪️ Polarité : `{row['Polarité estimée']}` — `{row['Véracité polarité']}`")

        if show_subtheme_table:
            # 📊 Export complet possible aussi
            st.markdown("### 📊 Moyenne des notes globales par sous-thème")
            afficher_dataframe_propre(filtered_stats, cmap="RdYlGn")


        if show_distribution_chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_melted["Score"].hist(bins=[1, 2, 3, 4, 5, 6], rwidth=0.8, align="left", ax=ax)
            ax.set_title("Distribution des notes globales (Score)")
            ax.set_xlabel("Note")
            ax.set_ylabel("Nombre de verbatims")
            st.pyplot(fig)

        def display_representative_verbatims(subtheme_label, label_text):
            st.markdown(f"#### {label_text}")
            subset = df_enriched[df_enriched[subtheme_label].notna()]
            subset = subset.sort_values("Note globale avis 1", ascending=(label_text == "💔 Négatif"))
            
            for i, v in subset.iterrows():
                st.markdown(f"- *{v['Verbatim public']}*")
                st.markdown(f"  ↪️ Note : **{v['Note globale avis 1']}**")
                try:
                    # Analyse IA brute
                    sentiment = analyser_sentiment(v["Verbatim complet"][:512])[0]
                    st.markdown(f"  🧠 IA : `{sentiment['label']}` (score={sentiment['score']:.2f})")
                except:
                    st.markdown("  ❌ Erreur d’analyse sentiment")

                st.markdown("---")

        if show_verbatims_examples:
            for name in best_subs.index:
                display_representative_verbatims(name, "\U0001F49A Positif")

            for name in worst_subs.index:
                display_representative_verbatims(name, "\U0001F494 Négatif")

        if show_profil_matrix:
            # 📈 Matrice Profil x Sous-thème
            col_cat = "Êtes-vous venu :"
            if col_cat in df_enriched.columns:
                df_temp = df_enriched[[col_cat] + subtheme_cols].copy()
                mat = df_temp.groupby(col_cat)[subtheme_cols].mean().T

                st.markdown("### 🔍 Matrice `Profil × Sous-thèmes`")
                afficher_dataframe_propre(mat.T, cmap="RdYlGn")  # .T si tu veux l’avoir en lignes

                fig, ax = plt.subplots(figsize=(min(15, len(mat.columns) * 1.5), len(mat) * 0.4 + 3))
                sns.heatmap(mat, cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Colonne de profil non trouvée.")

        st.markdown("### 🏅 Sous-thèmes les plus polarisants (avec polarité IA)")
        for name, row in best_subs.iterrows():
            st.markdown(f"- **{name}** — {int(row['count'])} verbatims")
            st.markdown(f"  ↪️ Polarité : {row['Polarité estimée']} — {row['Véracité polarité']}")

        # ---- Options d’affichage pour le graphe style maquette ----
        st.sidebar.markdown("### 🗺️ Graphe style maquette")
        show_corr_accel = st.sidebar.checkbox("Afficher Corrélation × Accélération", value=True)
        date_col_name = st.sidebar.text_input("Colonne de date (pour l'accélération)", value="Date avis")
        resample_freq = st.sidebar.selectbox("Granularité temps", ["W", "M", "D"], index=0)
        recent_k = st.sidebar.slider("Fenêtre 'premières occurrences' (périodes)", 1, 12, 3)

        # Nouveaux contrôles pour l’affichage
        use_interactive = st.sidebar.checkbox("Mode interactif (Plotly)", value=True)
        top_labels = st.sidebar.slider("Max labels visibles", 0, 50, 15)
        inclure_extremes = st.sidebar.checkbox("Toujours afficher les extrêmes", value=True)

        # ---- Calcul + affichage ----
        if show_corr_accel:
            if date_col_name not in df_enriched.columns:
                st.info(f"ℹ️ Colonne de date '{date_col_name}' absente : accélération fixée à 0 et pas d’alerte récente.")
                date_arg = None
            else:
                date_arg = date_col_name

            stats_bulles = calculer_bulles_corr_accel(
                df_enriched=df_enriched,
                subtheme_cols=subtheme_cols,
                note_col="Note globale avis 1",
                date_col=date_arg,
                freq=resample_freq,
                recent_k_periods=recent_k
            )

            st.markdown("### 🧭 Corrélation × Accélération (taille = % d’occurrences)")
            if use_interactive:
                tracer_bulles_corr_accel_plotly(stats_bulles, top_labels=top_labels, inclure_extremes=inclure_extremes)
            else:
                tracer_bulles_corr_accel(stats_bulles, top_labels=top_labels)





        # Export CSV - uniquement les sous-thèmes filtrés (vrai positifs/négatifs)
        cols_to_export = ["count", "mean", "Polarité estimée", "Véracité polarité"]
        st.download_button("⬇️ Télécharger les résultats (subthemes_filtrés.csv)",
            filtered_stats[cols_to_export].to_csv().encode("utf-8"),
            file_name="subthemes_filtrés.csv", mime="text/csv")

        # Optionnel : Export complet incluant les ambigus
        st.download_button("⬇️ Télécharger toutes les stats (incl. ambigus)",
            merged_stats[cols_to_export].to_csv().encode("utf-8"),
            file_name="subthemes_complets.csv", mime="text/csv")
    
        if st.button("📝 Générer un rapport de synthèse"):
            rapport = generer_rapport_openai(filtered_stats)
            if rapport:
                st.markdown("### 🧾 Rapport de synthèse")
                st.markdown(rapport)

                pdf_bytes = exporter_rapport_pdf(rapport)
                st.download_button(
                    label="⬇️ Télécharger le rapport en PDF",
                    data=pdf_bytes,
                    file_name="rapport_synthese_verbatims.pdf",
                    mime="application/pdf"
                )


