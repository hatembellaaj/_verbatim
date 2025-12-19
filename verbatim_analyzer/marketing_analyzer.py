import logging
import os
import re
import random

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


def associer_sous_themes_par_similarity(
    df_result,
    themes,
    text_col="Verbatim",
    model_name="all-MiniLM-L6-v2",
    seuil_similarite=0.5
):
    import numpy as np
    import logging
    from sentence_transformers import SentenceTransformer, util

    logging.info("➡️ Démarrage de l'association des sous-thèmes par similarité")
    logging.info(f"💠 Modèle utilisé : {model_name}")
    logging.info(f"🔎 Colonne de texte : {text_col}")
    logging.info(f"✪ Seuil de similarité : {seuil_similarite}")
    logging.info(f"📄 Dimensions du DataFrame : {df_result.shape}")

    if text_col not in df_result.columns:
        raise ValueError(f"❌ La colonne '{text_col}' est introuvable. Colonnes disponibles : {list(df_result.columns)}")

    # Chargement du modèle
    model = SentenceTransformer(model_name)
    verbatims = df_result[text_col].fillna("").astype(str).tolist()
    logging.info(f"🗣️ Nombre de verbatims à traiter : {len(verbatims)}")

    # --- 🧩 Préparation des textes à encoder ---
    enriched_texts, col_names = [], []

    for t in themes:
        theme_name = str(t.get("theme", t)).strip()

        # Cas 1 : le thème contient des sous-thèmes
        if isinstance(t, dict) and "subthemes" in t and t["subthemes"]:
            for s in t["subthemes"]:
                if isinstance(s, dict):
                    label = s.get("label", "").strip()
                    keywords = " ".join(s.get("keywords", []))
                    enriched_text = f"{label}. {keywords}" if keywords else label
                else:
                    label = str(s).strip()
                    enriched_text = label

                col_name = f"{theme_name}::{label}" if label else theme_name
                enriched_texts.append(enriched_text)
                col_names.append(col_name)

        # Cas 2 : thème sans sous-thème
        else:
            enriched_texts.append(theme_name)
            col_names.append(theme_name)

    logging.info(f"🧩 Textes de référence à encoder : {len(enriched_texts)}")
    if not enriched_texts:
        logging.warning("⚠️ Aucun thème ni sous-thème à encoder. Vérifie la structure de `themes`.")
        return df_result

    # --- 🔢 Encodage ---
    emb_verbatims = model.encode(verbatims, convert_to_tensor=True)
    emb_themes = model.encode(enriched_texts, convert_to_tensor=True)
    logging.info("✅ Encodage terminé")

    # --- 💫 Similarité ---
    scores = util.cos_sim(emb_verbatims, emb_themes).cpu().numpy()
    affectations = 0

    for idx, row in enumerate(scores):
        note = df_result.loc[idx, "Note globale avis 1"] if "Note globale avis 1" in df_result else 1
        for j, score in enumerate(row):
            if score >= seuil_similarite:
                df_result.loc[idx, col_names[j]] = note
                affectations += 1

    logging.info(f"✅ Association terminée — {affectations} affectations réalisées")
    return df_result


openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None


def extract_marketing_clusters_with_openai(
    texts_public,
    texts_private=None,
    nb_clusters=5,
    model_name="gpt-4o-mini",
    sample_size: int = 50,
    return_sample: bool = False,
):
    """
    Analyse les verbatims pour regrouper les avis en clusters marketing
    avec un prompt spécifique orienté note globale.

    Un échantillon aléatoire limité à ``sample_size`` verbatims est envoyé à
    l'API pour limiter le coût. Quand ``return_sample`` est à True, la liste
    réellement transmise est renvoyée pour affichage dans l'interface.
    """
    if not client:
        raise ValueError("Client OpenAI non initialisé.")

    if texts_private:
        verbatims_concat = [f"{pub} {priv}".strip() for pub, priv in zip(texts_public, texts_private)]
    else:
        verbatims_concat = texts_public

    available = len(verbatims_concat)
    effective_sample = min(max(1, sample_size), available) if available else 0
    indices = []
    if effective_sample and effective_sample < available:
        indices = random.sample(range(available), effective_sample)
        sampled_verbatims = [verbatims_concat[i] for i in indices]
    else:
        sampled_verbatims = verbatims_concat
        indices = list(range(len(sampled_verbatims)))

    logging.info(
        "📊 Échantillon OpenAI : %s verbatims sur %s disponibles",
        len(sampled_verbatims),
        available,
    )

    if not sampled_verbatims:
        raise ValueError("Aucun verbatim disponible pour l'extraction.")

    verbatims_joined = "\n".join(sampled_verbatims)

    prompt = f"""
Tu es un expert en marketing et en analyse de la satisfaction client.

Voici une liste d’avis clients (verbatims).

Ta tâche est de :
1. Identifier exactement {nb_clusters} *groupes thématiques* (clusters) d'expérience client.
2. Donner un **nom clair, court et fonctionnel** à chaque thème (ex : "Restauration", "Attractions", "Orientation").
3. Pour chaque thème, liste des **sous-thèmes** sous la forme d’objets enrichis avec deux champs :
   - `label` : une formulation explicite, claire et orientée (positive ou négative).
   - `keywords` : une liste de synonymes, variantes lexicales et expressions associées (issus du langage courant des visiteurs, incluant des formulations positives et négatives).

⚠️ Les sous-thèmes doivent être directement interprétables, actionnables et liés à une note positive ou négative.
⚠️ N’utilise pas de termes vagues ou neutres comme "ambiance agréable" ou "organisation".
⚠️ Les mots-clés doivent couvrir des variantes lexicales pour maximiser la reconnaissance par des modèles sémantiques (par ex. MiniLM, BERT).
⚠️ Chaque verbatim doit appartenir à au moins un des clusters générés.
⚠️ Si un verbatim ne rentre dans aucun thème existant, crée un nouveau thème/sous-thème pour le couvrir.

### 🧪 Exemple attendu :

Liste de verbatims :
- On a adoré le chemin lumineux dans la forêt, c'était magique.
- Il faisait nuit noire dans certaines zones, on ne voyait rien.
- On a attendu plus d'une heure à la première attraction.
- Il y avait des gens qui trichaient dans les files, c'était frustrant.
- Service très lent au snack, on a attendu 30 min pour un sandwich.
- Aucun panneau pour se repérer à l'entrée.
- Les enfants ont adoré l'attraction avec les dinosaures.

Exemple de Résultat attendu :
```python
[
  {{ "theme": "Attractions", "subthemes": [
      {{ "label": "Attractions puissantes et amusantes",
         "keywords": ["rollercoaster", "manège à sensations", "grands huit", "attractions fortes", "expérience amusante", "National 7", "attractions préférées des enfants"] }},
      {{ "label": "Manque d'attractions adaptées aux enfants",
         "keywords": ["attractions enfants", "manque pour petits", "pas assez de choix 1m20", "enfants déçus"] }}
  ]}},
  {{ "theme": "Expérience nocturne", "subthemes": [
      {{ "label": "Chemin enchanté illuminé et féerique",
         "keywords": ["spectacle lumineux", "chemin nocturne", "forêt magique", "lumières féeriques", "nocturne"] }},
      {{ "label": "Éclairage insuffisant la nuit",
         "keywords": ["nuit noire", "mauvais éclairage", "zones sombres", "on ne voit rien"] }}
  ]}}
]


```

Ne fournis aucune explication, uniquement la liste Python comme ci-dessus.

Liste des verbatims à analyser :
{verbatims_joined}
"""

    try:
        logging.info("📤 Envoi du prompt à OpenAI (%s)...", model_name)
        logging.debug("🧾 Prompt complet :\n%s", prompt)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.2,
        )
        usage = getattr(response, "usage", None)

        content = response.choices[0].message.content.strip()

        logging.debug("📩 Réponse brute du modèle :\n%s", content)

        cleaned = re.sub(r"```(?:python)?", "", content)
        cleaned = cleaned.replace("```", "").strip()

        logging.debug("🧹 Contenu nettoyé :\n%s", cleaned)

        themes = eval(cleaned)  # ou json.loads(cleaned)
        if not isinstance(themes, list):
            raise ValueError("La réponse n'est pas une liste de thèmes valide.")

        logging.info("📦 Thèmes extraits : %s", themes)

        if return_sample:
            return themes, sampled_verbatims, {
                "total": available,
                "sampled": len(sampled_verbatims),
                "indices": indices,
                "randomized": effective_sample < available,
            }, usage

        return themes

    except Exception as e:
        logging.exception("❌ Erreur lors de l'extraction des clusters marketing")
        raise

