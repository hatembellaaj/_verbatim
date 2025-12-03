import ast
import logging

logging.basicConfig(level=logging.INFO)

def nommer_clusters_openai_et_sous_clusters(client, df_result, nb_clusters=5):
    clusters = []
    subclusters = []

    # Concatène les verbatims de chaque cluster pour les envoyer à GPT
    for cluster_id in sorted(df_result['Cluster'].unique()):
        verbatims_cluster = df_result[df_result['Cluster'] == cluster_id]['Verbatim'].tolist()
        verbatims_extrait = "\n- " + "\n- ".join(verbatims_cluster[:20])

        prompt = f"""
Tu es un assistant d'analyse sémantique. Voici une liste de verbatims regroupés dans un même cluster :
{verbatims_extrait}

1. Donne un nom de thème représentatif pour ce groupe.
2. Propose 2 ou 3 sous-thèmes pertinents qui le composent.

Formate ta réponse ainsi :
{{"theme": "nom du thème", "subthemes": ["sous-thème 1", "sous-thème 2"]}}
Ne donne aucune explication. Retourne uniquement le dictionnaire.
"""

        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.strip()}]
            )
            content = completion.choices[0].message.content.strip()
            logging.info("\ud83d\udd39 Réponse OpenAI pour cluster %s : %s", cluster_id, content)

            result = ast.literal_eval(content)
            clusters.append(result["theme"])
            subclusters.append(", ".join(result["subthemes"]))
        except Exception as e:
            logging.warning("\u26a0\ufe0f Erreur lors du nommage du cluster %s : %s", cluster_id, str(e))
            clusters.append(f"Cluster {cluster_id}")
            subclusters.append("")

    return clusters, subclusters
