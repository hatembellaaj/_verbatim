# Plan de reconstruction en architecture hexagonale + React

Ce document décrit comment migrer l'application Streamlit actuelle vers une architecture hexagonale côté backend (FastAPI) et un frontend React.

## Côté backend (Python)

1. **Domaine** : formaliser les concepts clés `Verbatim`, `Analyse` et `Utilisateur`. Chaque module Streamlit existant fournit des comportements à transformer en cas d'usage.
2. **Cas d'usage** : exposer des services comme "créer un verbatim", "lister/filtrer", "analyser (marketing, IA, combinée)". Le fichier `application/use_cases.py` montre un exemple minimal.
3. **Ports/Adapters** :
   - Ports pour le stockage (base SQL/NoSQL), pour les moteurs d'IA (OpenAI, HF) et pour l'authentification.
   - Adapters dans `infrastructure/` pour concrétiser ces ports (ex : repository Postgres, service d'analytics utilisant `sentence-transformers` ou `openai`).
4. **API** : FastAPI sert d'adaptateur d'entrée HTTP. Le fichier `interfaces/api/main.py` expose déjà `/verbatims` et `/analysis`.
5. **Tests** : prioriser des tests unitaires sur le domaine et les cas d'usage (fixtures en mémoire), puis des tests d'intégration sur l'API.

## Côté frontend (React)

1. **Outiling** : Vite + TypeScript + React Query pour la data, React Router pour la navigation, Tailwind ou Mantine pour l'UI.
2. **Fonctionnalités à porter** :
   - Navigation principale (Marketing / IA Rating / Analyse combinée) via `src/app/routes.tsx`.
   - Composants de saisie de verbatim, affichage de tableaux/graphes (Chart.js ou Recharts) et génération de rapports.
   - Gestion utilisateurs (badge + admin) reliée à un provider d'auth (Auth0, Cognito ou backend custom).
3. **API client** : générer un client à partir du schéma OpenAPI du backend (outil `openapi-typescript`).
4. **Communication temps réel** : si nécessaire, utiliser Server Sent Events ou WebSocket pour les analyses longues.

## Migration progressive

- Démarrer par la parité fonctionnelle "Analyse combinée" (module `analyze_combined.py`) : extraire la logique d'analyse dans un service de domaine, l'exposer via FastAPI, puis créer un écran React pour saisir des verbatims et afficher les résultats.
- Enchaîner avec les parties Marketing et IA rating en réutilisant les mêmes patterns de ports/adapters.
- Remplacer progressivement l'UI Streamlit par la SPA React, tout en gardant la CLI/Streamlit comme outils internes si utile.
