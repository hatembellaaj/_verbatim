import streamlit as st
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import utils
from column_mapper import load_csv_with_mapping
from verbatim_analyzer.database import init_db
from verbatim_analyzer.marketing_analyzer import extract_marketing_clusters_with_openai, associer_sous_themes_par_similarity
from sidebar_options import get_sidebar_options
from report_utils import generer_et_afficher_rapport
from verbatim_analyzer.pricing import render_llm_selector


def run():
    st.header("📊 Analyse Marketing des verbatims")
    init_db()

    uploaded_file = st.file_uploader("📂 Téléverser un fichier CSV avec notes globales", type="csv")
    if uploaded_file is None:
        st.stop()

    df = load_csv_with_mapping(
        uploaded_file,
        required_fields=["Verbatim public", "Note globale avis 1"],
        optional_fields=["Verbatim privé"],
        key_prefix="marketing",
    )

    st.success(f"✅ Fichier chargé après mapping : {df.shape[0]} lignes")

    missing_required = [col for col in ["Verbatim public", "Note globale avis 1"] if col not in df.columns]
    if missing_required:
        st.error(
            "❌ Mapping incomplet : "
            + ", ".join(missing_required)
            + " est requis pour poursuivre."
        )
        st.stop()

    df["Verbatim complet"] = df["Verbatim public"].fillna("") + " " + df.get("Verbatim privé", "").fillna("")

    # === Options sidebar ===
    options = get_sidebar_options(uploaded_file)
    if options.get("use_openai"):
        st.sidebar.info(
            f"LLM sélectionné : **{options['llm_model']}**\n\n"
            f"Coût estimé : ${options['llm_input_cost']:.4f} /1k in · ${options['llm_output_cost']:.4f} /1k out"
        )

    with st.expander("⚙️ Choix du LLM & coûts OpenAI", expanded=options.get("use_openai", False)):
        chosen_model, in_cost, out_cost = render_llm_selector("OpenAI")
        options["llm_model"] = chosen_model
        options["llm_input_cost"] = in_cost
        options["llm_output_cost"] = out_cost

    # === Extraction des thèmes ===
    texts_public = df["Verbatim public"].astype(str).tolist()
    texts_private = df["Verbatim privé"].astype(str).tolist() if "Verbatim privé" in df.columns else [""] * len(df)

    themes = []
    if options["use_openai"]:
        with st.spinner("🔮 Extraction des clusters via OpenAI..."):
            try:
                themes = extract_marketing_clusters_with_openai(
                    texts_public,
                    texts_private,
                    options["nb_clusters"],
                    model_name=options["llm_model"],
                )
                st.success("✅ Clusters extraits avec succès")
                with st.expander("📂 Aperçu des thèmes extraits"):
                    for t in themes:
                        st.markdown(f"**{t['theme']}**")
                        for s in t["subthemes"]:
                            if isinstance(s, dict):
                                st.markdown(f"- {s['label']} _(keywords: {', '.join(s.get('keywords', []))})_")
                            else:
                                st.markdown(f"- {s}")
            except Exception as e:
                st.error(f"Erreur OpenAI : {e}")
                st.stop()
    else:
        user_themes = st.sidebar.text_area("Liste manuelle des clusters (JSON ou CSV)")
        if not user_themes.strip():
            st.warning("⚠️ Fournissez des thèmes manuels ou activez OpenAI.")
            st.stop()
        try:
            themes = pd.read_json(user_themes).to_dict(orient="records")
        except Exception:
            themes = [{"theme": t.strip(), "subthemes": []} for t in user_themes.split(",") if t.strip()]
        st.success(f"✅ {len(themes)} thèmes définis manuellement")

    if not themes:
        st.warning("⚠️ Aucun cluster défini.")
        st.stop()

    # === Attribution des sous-thèmes ===
    model_name = "all-MiniLM-L6-v2" if options["model_choice"] == "MiniLM" else "bert-base-nli-mean-tokens"
    df_enriched = associer_sous_themes_par_similarity(
        df, themes=themes, text_col="Verbatim complet", model_name=model_name, seuil_similarite=options["seuil_similarite"]
    )

    subtheme_cols = [c for c in df_enriched.columns if "::" in c]
    if not subtheme_cols:
        st.error("❌ Aucun sous-thème attribué.")
        st.stop()

    # === Matrice et incohérences ===
    if options["show_matrice_verbatims"]:
        matrice = utils.construire_matrice_verbatims(df_enriched, subtheme_cols)
        st.markdown("### 📐 Matrice Verbatims × Sous-thèmes")
        st.dataframe(matrice)

    if options["show_incoherences"]:
        incoherences = utils.verifier_coherence_semantique(df_enriched, subtheme_cols, seuil=0.3, alpha=0.7)
        st.markdown("### 🔍 Incohérences détectées")
        st.dataframe(incoherences)
        matrice_finale = utils.construire_matrice_finale(df_enriched, incoherences, subtheme_cols)
        st.markdown("### 📐 Matrice finale")
        st.dataframe(matrice_finale)

    # === Stats globales ===
    df_melted = df_enriched[["Note globale avis 1"] + subtheme_cols].melt(
        id_vars="Note globale avis 1", var_name="Sous-thème", value_name="Score"
    ).dropna()

    subtheme_stats = df_melted.groupby("Sous-thème")["Score"].agg(["count", "mean"]).sort_values("mean", ascending=False)

    # === Polarité ===
    df_polarite = utils.analyser_par_batches(df_enriched, batch_size=1000, process_func=utils.traiter_batch_polarite)
    merged_stats = subtheme_stats.merge(df_polarite, left_index=True, right_on="Sous-thème").set_index("Sous-thème")

    # === Meilleurs & pires sous-thèmes ===
    if options["show_best_worst"] and not merged_stats.empty:
        best = merged_stats.loc[merged_stats["mean"].idxmax()]
        worst = merged_stats.loc[merged_stats["mean"].idxmin()]
        st.success(f"🏅 Meilleur : **{best.name}** ({best['mean']:.2f})")
        st.error(f"💔 Pire : **{worst.name}** ({worst['mean']:.2f})")

    # === Tableau sous-thèmes filtrés ===
    if options["show_subtheme_table"]:
        st.markdown("### 📊 Moyenne des notes globales par sous-thème")
        st.dataframe(merged_stats[["count", "mean", "Polarité estimée", "Véracité polarité"]])

    # === Exemples de verbatims ===
    if options["show_verbatims_examples"]:
        st.markdown("### 📝 Exemples de verbatims par sous-thème")
        for col in subtheme_cols[:5]:
            st.markdown(f"**{col}**")
            subset = df_enriched[df_enriched[col].notna()].head(3)
            for _, row in subset.iterrows():
                st.markdown(f"- {row['Verbatim public']} (note {row['Note globale avis 1']})")

    # === Comparaison notes utilisées ===
    if options["show_note_comparison"]:
        note_distribution = df["Note globale avis 1"].value_counts().sort_index()
        used_notes = df_melted["Note globale avis 1"].value_counts().sort_index()
        used_unique = df_enriched.loc[df_enriched[subtheme_cols].notna().any(axis=1), "Note globale avis 1"].value_counts().sort_index()
        note_comparison = pd.DataFrame({
            "Total verbatims": note_distribution,
            "Utilisés (associations)": used_notes,
            "Verbatims uniques utilisés": used_unique
        }).fillna(0).astype(int)
        st.markdown("### 🧾 Comparaison notes utilisées")
        st.dataframe(note_comparison)

    # === Graphe corrélation × accélération ===
    if options["show_corr_accel"]:
        stats_bulles = utils.calculer_bulles_corr_accel(df_enriched, subtheme_cols, note_col="Note globale avis 1")
        fig = utils.tracer_bulles_corr_accel_plotly(stats_bulles)
        st.plotly_chart(fig, use_container_width=True)

    # === Rapport IA ===
    if st.button("📝 Générer un rapport de synthèse"):
        generer_et_afficher_rapport(
            merged_stats,
            titre="Rapport de synthèse Marketing",
            filename="rapport_synthese_marketing.pdf"
        )

    # === Export CSV final ===
    st.subheader("⬇️ Export des résultats")
    csv_bytes = utils.preparer_csv_export(df_enriched, "resultats_marketing.csv")
    st.download_button("Télécharger résultats", data=csv_bytes, file_name="resultats_marketing.csv", mime="text/csv")
