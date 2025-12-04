import streamlit as st


def render_manual() -> None:
    st.title("📘 Manuel d'utilisation")
    st.markdown(
        """
        Bienvenue dans le manuel d'utilisation de l'application d'analyse des verbatims client.
        Cette page rassemble les étapes essentielles pour vous authentifier, configurer vos analyses
        et interpréter les résultats.
        """
    )

    st.header("1. Connexion et sécurité")
    st.markdown(
        """
        - **Connexion** : saisissez votre identifiant et votre mot de passe dans le panneau de gauche.
        - **Déconnexion** : utilisez le bouton dédié dans la barre latérale.
        - **Changement de mot de passe** : ouvrez l'onglet « 🔑 Modifier mon mot de passe »
          pour saisir votre mot de passe actuel puis définir un nouveau mot de passe.
        - **Gestion des utilisateurs (administrateurs)** : les comptes administrateurs peuvent créer
          de nouveaux utilisateurs et définir leur rôle directement depuis la barre latérale.
        """
    )

    st.header("2. Navigation dans l'application")
    st.markdown(
        """
        - Le menu latéral vous permet de passer d'un module à l'autre : **Marketing**, **IA Rating**
          et **Analyse combinée**.
        - Utilisez le sélecteur de navigation en haut à droite pour revenir à l'application ou
          revenir sur ce manuel à tout moment.
        """
    )

    st.header("3. Analyse combinée des verbatims")
    st.markdown(
        """
        - **Importer un fichier CSV** depuis l'étape 1 ; la colonne « Verbatim public » est requise.
        - **Choisir le mode d'analyse** :
          - *Analyse Marketing* pour travailler sur la note client.
          - *Analyse IA* pour utiliser une note générée automatiquement.
        - **Options d'affichage** : personnalisez les graphiques, tableaux et matrices via la barre
          latérale (comparaison des verbatims, histogrammes des scores, matrices de profils, etc.).
        - **Définition des thèmes** :
          - Activez OpenAI pour extraire automatiquement les clusters, ou fournissez vos thèmes
            manuellement.
          - Ajoutez, renommez ou enrichissez les thèmes et sous-thèmes grâce aux formulaires dédiés.
        - **Visualisation** : explorez l'arborescence des clusters détectés, les distributions de
          scores et les exemples de verbatims représentatifs.
        """
    )

    st.header("4. Export et rapports")
    st.markdown(
        """
        - Générez des rapports détaillés en utilisant les boutons d'export disponibles dans les
          modules d'analyse.
        - Les rapports incluent les données filtrées selon vos options d'affichage, ce qui vous
          permet de partager des résultats cohérents avec vos sélections.
        """
    )

    st.header("5. Gestion des incohérences")
    st.markdown(
        """
        - Surveillez les alertes lors de l'import : colonnes manquantes, types inattendus ou
          encodage erroné sont souvent la source d'incohérences. Corrigez le fichier puis
          réimportez-le.
        - Comparez les notes Marketing et IA lorsque les deux sont disponibles. Une divergence
          significative signale un jeu de données à nettoyer ou un paramétrage de scoring à revoir.
        - En cas de clusters surprenants, supprimez les doublons, harmonisez la casse et les
          accents, puis relancez l'extraction automatique ou ajustez vos thèmes manuels avant de
          poursuivre l'analyse.
        - Lorsque vous exportez un rapport, consignez les incohérences observées (lignes supprimées,
          colonnes corrigées, règles d'exclusion) pour garder un historique clair des corrections.
        - Si une incohérence persiste, recommencez avec un sous-échantillon de verbatims pour
          identifier la ligne ou la colonne problématique, puis réexécutez l'analyse complète.
        """
    )

    st.header("6. Conseils pratiques")
    st.markdown(
        """
        - Limitez la taille des fichiers CSV pour accélérer le chargement et l'affichage des
          graphiques.
        - En cas de modification des thèmes ou sous-thèmes, pensez à ré-exécuter l'extraction pour
          mettre à jour les résultats.
        - Si vous rencontrez un problème, revenez à cette page via le menu en haut à droite pour
          vérifier les étapes essentielles.
        """
    )
