import streamlit as st

from verbatim_analyzer.pricing import format_cost, get_model_cost, load_pricing

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
    pricing = load_pricing()
    model_choices = sorted(pricing.keys()) or ["gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]

    default_model = st.session_state.get("llm_model") or ("gpt-4o-mini" if "gpt-4o-mini" in model_choices else model_choices[0])
    llm_model = st.sidebar.selectbox("Modèle LLM (OpenAI)", model_choices, index=model_choices.index(default_model) if default_model in model_choices else 0)

    default_input_cost, default_output_cost = get_model_cost(llm_model, pricing)
    input_cost = st.sidebar.number_input(
        "Coût entrée / 1k tokens ($)",
        min_value=0.0,
        value=float(default_input_cost or 0.0),
        format="%.6f",
    )
    output_cost = st.sidebar.number_input(
        "Coût sortie / 1k tokens ($)",
        min_value=0.0,
        value=float(default_output_cost or 0.0),
        format="%.6f",
    )

    st.sidebar.markdown(
        f"**Tarif tokens :** {format_cost(input_cost, output_cost)}\n\n"
        "Mettre à jour via l'API de pricing ou saisie manuelle."
    )

    options.update({
        "use_openai":       st.sidebar.checkbox("Utiliser OpenAI pour les clusters"),
        "nb_clusters":      st.sidebar.slider("Nombre de clusters", min_value=2, max_value=1000, value=5),
        "model_choice":     st.sidebar.radio("Modèle d'encodage", ["MiniLM", "BERT"]),
        "seuil_similarite": st.sidebar.slider("Seuil de similarité (MiniLM/BERT)", 0.0, 1.0, 0.45, step=0.05),
        "llm_model":        llm_model,
        "llm_input_cost":   input_cost,
        "llm_output_cost":  output_cost,
    })

    st.session_state["llm_model"] = llm_model
    st.session_state["llm_pricing"] = {
        "input": input_cost,
        "output": output_cost,
        "source": "manual" if (input_cost != (default_input_cost or 0.0) or output_cost != (default_output_cost or 0.0)) else "file",
    }

    return options
