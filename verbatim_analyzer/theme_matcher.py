import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from verbatim_analyzer.utils import get_embedding_model, compute_embeddings

# Exemple de thèmes prédéfinis (à adapter)
THEMES = [
    "Temps d'attente trop long",
    "Propreté des lieux",
    "Accueil du personnel",
    "Application mobile difficile à utiliser",
    "Ambiance générale agréable"
]


def match_verbatims_to_themes(verbatims, model_name="MiniLM"):
    model = get_embedding_model(model_name)

    theme_embeddings = compute_embeddings(THEMES, model)
    verbatim_embeddings = compute_embeddings(verbatims, model)

    matches = []
    for vec in verbatim_embeddings:
        sims = cosine_similarity([vec], theme_embeddings)[0]
        best_idx = int(np.argmax(sims))
        matches.append((THEMES[best_idx], float(np.max(sims))))

    return matches
