import streamlit as st

def get_sidebar_options(uploaded_file=None):
    """Construit la sidebar et retourne un dictionnaire d'options."""
    st.sidebar.markdown("### ⚙️ Options d'affichage")

    is_big_file = uploaded_file and hasattr(uploaded_file, "size") and uploaded_file.size > 4_000_000

    options = {
        "show_note_comparison":    st.sidebar.checkbox("Afficher comparaison des verbatims utilisés", value=not is_big_file),
        "show_subtheme_table":     st.sidebar.checkbox("Afficher tableau sous-thèmes filtrés", value=True),
        "show_best_worst":         st.sidebar.checkbox("Afficher meilleurs/pires sous-thèmes", value=not is_big_file),
        "show_distribution_chart": st.sidebar.checkbox("Afficher histogramme des scores", value=not is_big_file),
        "show_verbatims_examples": st.sidebar.checkbox("Afficher verbatims représentatifs", value=False),
        "show_profil_matrix":      st.sidebar.checkbox("Afficher matrice Profil × Sous-thèmes", value=not is_big_file),
        "show_matrice_verbatims":  st.sidebar.checkbox("Afficher matrice Verbatims × Sous-clusters", value=False),
        "show_unassigned":         st.sidebar.checkbox("Afficher les verbatims non associés", value=False),
        "show_incoherences":       st.sidebar.checkbox("Afficher incohérences sémantiques", value=False),
        # ✅ Ajout manquant
        "show_corr_accel":         st.sidebar.checkbox("Afficher graphe Corrélation × Accélération", value=True),
    }

    st.sidebar.markdown("### ⚙️ Paramètres Clusters")
    options.update({
        "use_openai":       st.sidebar.checkbox("Utiliser OpenAI pour les clusters"),
        "nb_clusters":      st.sidebar.slider("Nombre de clusters", min_value=2, max_value=1000, value=5),
        "model_choice":     st.sidebar.radio("Modèle d'encodage", ["MiniLM", "BERT"]),
        "seuil_similarite": st.sidebar.slider("Seuil de similarité (MiniLM/BERT)", 0.0, 1.0, 0.45, step=0.05),
    })

    return options
