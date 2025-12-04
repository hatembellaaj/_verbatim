# Backend hexagonal skeleton

Ce dossier propose un squelette minimal pour réécrire l'application en Python avec une architecture hexagonale.

## Structure proposée

- `domain/` : modèles métier et ports (interfaces des dépendances externes).
- `application/` : cas d'usage orchestrant le domaine et les ports.
- `infrastructure/` : implémentations concrètes (persistances, services IA, adapters).
- `interfaces/api/` : adaptateur d'entrée HTTP (FastAPI) exposant les cas d'usage.

## Lancer l'API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn interfaces.api.main:app --reload
```

## Cas d'usage inclus

- Création et listage des verbatims (stockage en mémoire pour l'instant).
- Déclenchement d'une analyse combinée (appel d'un service d'analyse simulé).

Les modules sont volontairement simples pour servir de point d'accroche à la migration du code Streamlit existant (par ex. `marketing.py`, `ia_rating.py`, `analyze_combined.py`).
