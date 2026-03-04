import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import plotly.express as px
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Initialisation OpenAI
client = OpenAI()
_PRICING_CACHE = None

# Chargement du modèle RoBERTa (multilingue) pour le sentiment
analyser_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Config TF-IDF
USE_TFIDF = True
TFIDF_ALPHA = 0.3

# Listes de mots pour règles heuristiques
REGEX_MOTS_NEGATIFS = re.compile(
    r"\b(manque|nul|horrible|catastrophique|décevant|trop long|attente interminable|mal organisé|inutile|déplorable|pas\s+(terrible|bon|correct|agréable)|rien de spécial|très cher|arnaque)\b",
    flags=re.IGNORECASE
)
MOTS_NEGATIFS = ["manque", "insuffisant", "mauvais", "problème", "attente", "cher", "décevant", "raté"]
MOTS_POSITIFS = ["magique", "féérique", "excellent", "bravo", "super", "parfait"]

MALE_FIRSTNAMES = {
    "adam", "alexandre", "antoine", "arthur", "benjamin", "charles", "david", "enzo",
    "gabriel", "hugo", "jacques", "jean", "jules", "kevin", "leo", "lucas", "louis",
    "mathieu", "maxime", "mehdi", "michel", "nicolas", "noah", "paul", "pierre", "raphael",
    "thomas", "victor", "yassine"
}

FEMALE_FIRSTNAMES = {
    "alice", "amelie", "anais", "camille", "charlotte", "chloe", "claire", "emma", "eva",
    "fatima", "ines", "isabelle", "jade", "julie", "laura", "lea", "lina", "louise", "manon", "marie",
    "mila", "nadia", "nina", "oceane", "pauline", "sarah", "sophie", "zoe"
}


# -----------------------------
# CLUSTERING
# -----------------------------
from verbatim_analyzer.marketing_analyzer import (
    extract_marketing_clusters_with_openai,
    associer_sous_themes_par_similarity
)


# -----------------------------
# MATRICES & COHERENCE
# -----------------------------
def construire_matrice_verbatims(df_enriched, subtheme_cols, note_col="Note globale avis 1"):
    matrice = pd.DataFrame(index=df_enriched.index)
    for col in subtheme_cols:
        matrice[col] = np.where(df_enriched[col].notna(), df_enriched[note_col], np.nan)
    matrice.insert(0, "Verbatim complet", df_enriched["Verbatim complet"])
    matrice.insert(1, "Note globale", df_enriched[note_col])
    return matrice


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
    logging.info(f"📐 Matrice finale construite : {matrice.shape[0]} lignes × {matrice.shape[1]} colonnes")
    return matrice


def verifier_coherence_semantique(df, subtheme_cols, seuil=0.3, alpha=0.9, model_name="all-MiniLM-L6-v2"):
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

    # TF-IDF
    if USE_TFIDF:
        texts = verbatims + subthemes
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(texts)
        verbatim_vecs = X[:len(verbatims)]
        subtheme_vecs = X[len(verbatims):]
        sims_tfidf = [cosine_similarity(verbatim_vecs[i], subtheme_vecs[i])[0][0] for i in range(len(verbatims))]
    else:
        sims_tfidf = [0.0] * len(verbatims)

    # MiniLM
    model = SentenceTransformer(model_name)
    emb_verbatims = model.encode(verbatims, convert_to_tensor=True)
    emb_subthemes = model.encode(subthemes, convert_to_tensor=True)
    sims_minilm = util.cos_sim(emb_verbatims, emb_subthemes).diagonal().cpu().numpy().tolist()

    # Fusion
    sims_final = [alpha * s_minilm + (1 - alpha) * s_tfidf
                  for s_minilm, s_tfidf in zip(sims_minilm, sims_tfidf)]

    assignments["Similarité_TFIDF"] = sims_tfidf
    assignments["Similarité_MiniLM"] = sims_minilm
    assignments["Score_final"] = sims_final

    # Décision
    def coherence_note_mot(note, sous_theme):
        st_lower = sous_theme.lower()
        if note >= 5 and any(m in st_lower for m in MOTS_NEGATIFS):
            return False
        if note <= 2 and any(m in st_lower for m in MOTS_POSITIFS):
            return False
        return True

    decisions = []
    for i, row in assignments.iterrows():
        note = row["Note globale"]
        sous_theme = row["Sous-thème"]
        s_tfidf = row["Similarité_TFIDF"]
        score = row["Score_final"]

        if s_tfidf < 0.03:
            decisions.append("Non retenue")
        elif not coherence_note_mot(note, sous_theme):
            decisions.append("Non retenue")
        elif score < seuil:
            decisions.append("⚠️ Suspect")
        else:
            decisions.append("OK")

    assignments["Decision"] = decisions
    return assignments


# -----------------------------
# POLARITE & SENTIMENT
# -----------------------------
def contient_mots_negatifs(texte):
    return bool(REGEX_MOTS_NEGATIFS.search(texte))


def _normaliser_texte(texte: str) -> str:
    if not isinstance(texte, str):
        return ""
    sans_accents = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", sans_accents.lower())


def _normaliser_sexe(valeur) -> str:
    if not isinstance(valeur, str) or not valeur.strip():
        return ""
    nettoyee = _normaliser_texte(valeur)
    if nettoyee in {"f", "femme", "feminin", "female", "woman"}:
        return "Femme"
    if nettoyee in {"h", "homme", "masculin", "male", "man"}:
        return "Homme"
    return ""


def inferer_sexe_par_prenom(prenom) -> str:
    """Infère un sexe probable à partir du 1er prénom normalisé.

    Exemple: "Isabelle" → "Femme" (présent dans FEMALE_FIRSTNAMES).
    """
    if not isinstance(prenom, str) or not prenom.strip():
        return ""
    prenom_canonique = _normaliser_texte(prenom.split()[0])
    if prenom_canonique in FEMALE_FIRSTNAMES:
        return "Femme"
    if prenom_canonique in MALE_FIRSTNAMES:
        return "Homme"
    return ""


def enrichir_colonnes_demographiques(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalise la colonne Sexe et complète les valeurs manquantes via Prénom."""
    if "Sexe" not in df.columns and "Prénom" not in df.columns:
        return df, 0

    if "Sexe" not in df.columns:
        df["Sexe"] = ""

    sexe_normalise = df["Sexe"].apply(_normaliser_sexe)
    sexe_inferre = df["Prénom"].apply(inferer_sexe_par_prenom) if "Prénom" in df.columns else ""

    mask_manquant = sexe_normalise == ""
    if isinstance(sexe_inferre, pd.Series):
        sexe_normalise = sexe_normalise.mask(mask_manquant, sexe_inferre)

    sexe_normalise = sexe_normalise.replace("", "Inconnu")
    lignes_inferrees = int((mask_manquant & (sexe_normalise != "Inconnu")).sum())

    df["Sexe"] = sexe_normalise
    return df, lignes_inferrees


def analyser_sentiment_mixte(texte):
    try:
        texte = texte.strip()
        if not texte:
            return {"label": "Neutral", "score": 0.0, "source": "vide"}
        res = analyser_sentiment(texte[:512])[0]
        score_label = res['label']
        score = int(score_label.split()[0])
        negatif_detecte = contient_mots_negatifs(texte)
        if score >= 4 and negatif_detecte:
            return {"label": "Mixed (positif mais critique)", "score": score, "source": "mixte"}
        elif score <= 2 and not negatif_detecte:
            return {"label": "Mixed (note faible mais pas critique)", "score": score, "source": "mixte"}
        elif negatif_detecte and score >= 3:
            return {"label": "Negative", "score": score, "source": "regex négatif"}
        else:
            return {"label": "Positive" if score >= 4 else "Negative" if score <= 2 else "Neutral",
                    "score": score, "source": "RoBERTa"}
    except Exception as e:
        return {"label": "Erreur", "score": 0, "source": str(e)}


def verifier_polarite_openai(sous_theme: str, polarite_estimee: str, note_moyenne: float, model: str | None = None) -> str:
    prompt = f"""
    Voici un sous-thème client : "{sous_theme}"
    Polarité détectée automatiquement : {polarite_estimee}
    Note moyenne associée : {note_moyenne}
    Corrige la polarité en tenant compte du sens réel du sous-thème.
    Réponds uniquement par un mot : Positive, Negative ou Neutre.
    """
    try:
        chosen_model = model
        if chosen_model is None:
            try:
                import streamlit as st  # lazy import pour récupérer le choix utilisateur si dispo
                chosen_model = st.session_state.get("llm_model")
            except Exception:
                chosen_model = None

        chosen_model = chosen_model or "gpt-4o-mini"

        prompt_tokens_estimes = _estimer_tokens(prompt)
        cout_estime = _estimer_cout(chosen_model, prompt_tokens_estimes, 50)
        logging.info(
            f"💰 Estimation coût OpenAI ({chosen_model}) — prompt: {prompt_tokens_estimes} tokens, "
            f"réponse estimée: 50 tokens → ~{cout_estime:.6f}$"
        )

        response = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        _log_cout_reel(chosen_model, response)

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


def analyser_par_batches(df, batch_size=1000, process_func=None):
    if process_func is None:
        raise ValueError("Vous devez fournir une fonction process_func(batch_df)")
    total = len(df)
    results = []
    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i+batch_size].copy()
        try:
            result = process_func(batch_df)
            if not result.empty:
                results.append(result)
        except Exception as e:
            logging.error(f"Erreur dans le batch {i//batch_size + 1} : {e}")
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Sous-thème", "Polarité estimée", "Véracité polarité"])


def traiter_batch_polarite(batch_df):
    subtheme_cols = [col for col in batch_df.columns if "::" in col]
    if not subtheme_cols:
        return pd.DataFrame(columns=["Sous-thème", "Polarité estimée", "Véracité polarité"])
    df_melted_batch = batch_df[["Note globale avis 1"] + subtheme_cols].melt(
        id_vars="Note globale avis 1",
        var_name="Sous-thème",
        value_name="Score"
    ).dropna()
    return enrichir_polarite_veracite(df_melted_batch, batch_df, subtheme_cols)


def enrichir_polarite_veracite(df_melted, df_enriched, subtheme_cols):
    polarite_info = []
    for subtheme in subtheme_cols:
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
        if contient_mots_negatifs(subtheme) and polarite == "Positive":
            polarite = "Neutre"
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
    return pd.DataFrame(polarite_info)


def _load_pricing():
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE
    pricing_file = Path(__file__).resolve().parent.parent / "openai_pricing.json"
    try:
        with pricing_file.open() as f:
            _PRICING_CACHE = json.load(f)
    except Exception as e:
        logging.warning(f"⚠️ Impossible de lire le fichier de pricing OpenAI ({pricing_file}): {e}")
        _PRICING_CACHE = {}
    return _PRICING_CACHE


def _estimer_tokens(texte: str) -> int:
    # Estimation simple : ~4 caractères par token (approx GPT-3.5/4)
    return max(1, len(texte) // 4)


def _estimer_cout(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _load_pricing()
    if model not in pricing:
        logging.info(f"ℹ️ Pas de tarif pour {model} → coût non estimé")
        return 0.0
    p = pricing[model]
    input_price = p.get("input_per_1k_tokens", 0)
    output_price = p.get("output_per_1k_tokens", 0)
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


def _log_cout_reel(model: str, response):
    usage = getattr(response, "usage", None)
    if not usage:
        logging.info("ℹ️ Pas de détail d'usage OpenAI retourné ; impossible de journaliser le coût réel.")
        return
    input_tokens = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens")
    output_tokens = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens")
    total_tokens = getattr(usage, "total_tokens", None) or usage.get("total_tokens")
    cout = _estimer_cout(model, input_tokens or 0, output_tokens or 0)
    logging.info(
        f"💳 Coût OpenAI réel ({model}) — prompt: {input_tokens} tokens, réponse: {output_tokens} tokens, "
        f"total: {total_tokens} tokens → ~{cout:.6f}$"
    )


# -----------------------------
# STATS & VISUALISATION
# -----------------------------
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


def calculer_bulles_corr_accel(df_enriched, subtheme_cols, note_col="Note globale avis 1",
                               date_col: Optional[str] = "Date avis", freq="W", recent_k_periods=3):
    df = df_enriched.copy()
    y = pd.to_numeric(df[note_col], errors="coerce")
    total_occ = df[subtheme_cols].notna().sum().sum()
    prop = {c: (df[c].notna().sum() / total_occ) if total_occ else 0.0 for c in subtheme_cols}

    corr = {}
    for c in subtheme_cols:
        x = df[c].notna().astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 3 and x[m].std() > 0 and y[m].std() > 0:
            corr[c] = float(np.corrcoef(x[m], y[m])[0, 1])
        else:
            corr[c] = 0.0

    accel, first_seen, recent_first = {}, {}, {}
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        now = (df[date_col].max() or pd.Timestamp.utcnow())
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


def tracer_bulles_corr_accel_plotly(stats: pd.DataFrame, top_labels: int = 15, inclure_extremes: bool = True):
    if stats.empty:
        return
    data = stats.reset_index().rename(columns={"Sous-thème":"Sous-theme"})
    data["Catégorie"] = np.where(data["is_recent_first"], "Premières occurrences récentes", "Occurrences établies")
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
        hover_name="Sous-theme",
        hover_data={"corr":":.2f", "accel":":.2f", "prop":":.1%", "first_seen": True, "Sous-theme": False}
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.add_shape(type="rect", x0=-1, x1=0, y0=-1e9, y1=1e9, fillcolor="#f8d7da", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=-1e9, y1=1e9, fillcolor="#d4edda", opacity=0.25, line_width=0)
    return fig


def tracer_bulles_corr_accel(stats: pd.DataFrame, title: str = "Corrélation × Accélération des sous-thèmes", top_labels: int = 20):
    if stats.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    sizes = (stats["prop"] * 7000).clip(lower=60)
    face_colors = np.where(stats["is_recent_first"], "#ffb74d", "#f2f2f2")
    edge_colors = "#7C2A90"
    ax.scatter(stats["corr"], stats["accel"], s=sizes, c=face_colors, edgecolors=edge_colors, linewidths=2, alpha=0.95)
    for name in stats.head(top_labels).index:
        ax.annotate(name, (stats.loc[name, "corr"], stats.loc[name, "accel"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    return fig


# -----------------------------
# EXPORTS
# -----------------------------
def preparer_csv_export(df, filename):
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        logging.exception("Erreur pendant l'export CSV")
        return b""


# -----------------------------
# AUTOMATIC : NOTE IA
# -----------------------------
def calculer_note_ia(verbatim: str) -> int:
    """
    Calcule une note IA (1–5) à partir d'un verbatim.
    1. Génère une note brute via BERT multilingue.
    2. Ajuste la note si des critiques sont détectées dans un avis trop positif.
    3. Fallback neutre = 3 en cas d'erreur.
    """
    try:
        texte = str(verbatim).strip()
        if not texte:
            return 3

        # Étape 1 : Note brute via BERT
        res = analyser_sentiment(texte[:512])[0]
        score_label = res['label']
        note_brute = int(score_label.split()[0])  # ex: "5 stars"

        # Étape 2 : Ajustement heuristique avec regex
        # On cherche des signaux de critiques même dans un avis positif
        critiques_regex = re.compile(
            r"(manque|dommage|pas assez|probl[eè]me|trop cher|décevant|attente|"
            r"nul|rat[ée]|osez habiller|nous manquer|manquera|supprim[ée]|fermeture)",
            flags=re.IGNORECASE
        )

        if note_brute == 5 and critiques_regex.search(texte):
            logging.info(f"⬇️ Note IA ajustée de 5 → 4 (critiques détectées dans '{texte[:80]}...')")
            return 4

        return note_brute

    except Exception as e:
        logging.error(f"Erreur calcul note IA: {e}")
        return 3  # neutre fallback
