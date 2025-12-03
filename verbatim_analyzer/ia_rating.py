import streamlit as st
import pandas as pd
import logging
import plotly.express as px 
import utils
from sidebar_options import get_sidebar_options
from report_utils import generer_et_afficher_rapport


def run():
    st.header("🤖 Analyse basée sur note IA")

    uploaded_file = st.file_uploader("📂 Téléverser un fichier CSV", type="csv")
    if uploaded_file is None:
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier chargé : {len(df)} lignes")
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()

    if "Verbatim public" not in df.columns:
        st.error("❌ Le fichier doit contenir une colonne 'Verbatim public'.")
        st.stop()

    df["Verbatim complet"] = df["Verbatim public"].fillna("") + " " + df.get("Verbatim privé", "").fillna("")

    # === Options sidebar ===
    options = get_sidebar_options(uploaded_file)

    # === Étape 1 : Calcul note IA ===
    st.subheader("🧠 Génération de la note IA (1–5)")
    df["Note IA"] = df["Verbatim public"].astype(str).apply(utils.calculer_note_ia)
    st.success("✅ Notes IA générées")

    # === Étape 2 : Attribution sous-thèmes ===
    model_name = "all-MiniLM-L6-v2" if options["model_choice"] == "MiniLM" else "bert-base-nli-mean-tokens"

    if options["use_openai"]:
        st.warning("⚠️ Extraction automatique des clusters avec OpenAI pas encore intégrée ici.")
        st.stop()
    else:
        user_themes = st.sidebar.text_area("Thèmes (JSON ou CSV)").strip()
        if not user_themes:
            st.warning("⚠️ Aucun thème fourni. Utilisez OpenAI ou saisissez une liste.")
            st.stop()
        try:
            themes = pd.read_json(user_themes).to_dict(orient="records")
        except Exception:
            themes = [{"theme": t.strip(), "subthemes": []} for t in user_themes.split(",") if t.strip()]

    df_enriched = utils.associer_sous_themes_par_similarity(
        df, themes=themes, text_col="Verbatim complet", model_name=model_name, seuil_similarite=options["seuil_similarite"]
    )

    subtheme_cols = [c for c in df_enriched.columns if "::" in c]
    if not subtheme_cols:
        st.warning("⚠️ Aucun sous-thème détecté. Vérifiez vos paramètres.")
        st.stop()

    # === Matrice et incohérences ===
    if options["show_matrice_verbatims"]:
        matrice = utils.construire_matrice_verbatims(df_enriched, subtheme_cols, note_col="Note IA")
        st.markdown("### 📐 Matrice Verbatims × Sous-thèmes (note IA)")
        st.dataframe(matrice)

    if options["show_incoherences"]:
        incoherences = utils.verifier_coherence_semantique(df_enriched, subtheme_cols, seuil=0.3, alpha=0.7)
        st.markdown("### 🔍 Incohérences détectées")
        st.dataframe(incoherences)

    # === Polarité & stats ===
    st.subheader("📊 Analyse Polarité & Véracité")
    df_polarite = utils.analyser_par_batches(df_enriched, batch_size=1000, process_func=utils.traiter_batch_polarite)
    if df_polarite.empty:
        st.warning("⚠️ Aucune polarité détectée.")
    else:
        st.dataframe(df_polarite)

    # === Statistiques globales sur les clusters ===
    st.subheader("📈 Statistiques de répartition des clusters")

    cluster_counts = (
        df_enriched[subtheme_cols].notna().sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Sous-thème", 0: "Occurrences"})
    )

    if cluster_counts.empty:
        st.warning("⚠️ Aucun cluster détecté pour générer des statistiques.")
    else:
        # 1. Affichage chiffres bruts
        st.markdown("### 🔢 Nombre de verbatims par sous-thème")
        st.dataframe(cluster_counts)   # ✅ corrigé

        # 2. Camembert (plotly)
        fig_pie = px.pie(
            cluster_counts,
            names="Sous-thème",
            values="Occurrences",
            title="Répartition des verbatims par sous-thème"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 3. Histogramme (plotly)
        fig_bar = px.bar(
            cluster_counts,
            x="Sous-thème",
            y="Occurrences",
            title="Histogramme des verbatims par sous-thème"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 4. Histogramme des notes IA par cluster
        df_melted = df_enriched[["Note IA"] + subtheme_cols].melt(
            id_vars="Note IA", var_name="Sous-thème", value_name="Assoc"
        ).dropna()
        fig_hist = px.histogram(
            df_melted,
            x="Note IA",
            color="Sous-thème",
            barmode="overlay",
            nbins=5,
            title="Distribution des notes IA par sous-thème"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


        # === Statistiques de répartition des clusters par profil ===
        if "Êtes-vous venu :" in df_enriched.columns:
            st.subheader("👥 Répartition des clusters par profil")

            # Préparation données longues
            df_melted_profile = df_enriched[["Êtes-vous venu :"] + subtheme_cols].melt(
                id_vars="Êtes-vous venu :", var_name="Sous-thème", value_name="Assoc"
            ).dropna()

            # 1. Tableau croisé
            table_profil = (
                df_melted_profile
                .groupby(["Êtes-vous venu :", "Sous-thème"])["Assoc"]
                .count()
                .reset_index()
                .pivot(index="Êtes-vous venu :", columns="Sous-thème", values="Assoc")
                .fillna(0)
                .astype(int)
            )
            st.markdown("### 📊 Tableau de répartition par profil")
            st.dataframe(table_profil)

            # 2. Histogramme groupé
            fig_bar_profil = px.bar(
                df_melted_profile,
                x="Sous-thème",
                y="Assoc",
                color="Êtes-vous venu :",
                barmode="group",
                title="Histogramme : répartition des verbatims par sous-thème et profil"
            )
            st.plotly_chart(fig_bar_profil, use_container_width=True)

            # 3. Camemberts par profil
            st.markdown("### 🥧 Camemberts par profil")
            for profil in df_melted_profile["Êtes-vous venu :"].unique():
                subset = df_melted_profile[df_melted_profile["Êtes-vous venu :"] == profil]
                fig_pie_profil = px.pie(
                    subset,
                    names="Sous-thème",
                    values="Assoc",
                    title=f"Répartition des verbatims ({profil})"
                )
                st.plotly_chart(fig_pie_profil, use_container_width=True)
        else:
            st.info("ℹ️ Aucun champ 'Êtes-vous venu :' trouvé dans vos données.")



    # === Meilleurs & pires sous-thèmes ===
    if options["show_best_worst"] and not df_polarite.empty:
        best = df_polarite.loc[df_polarite["Véracité polarité"].isin(["✅ Vrai positif", "✅ Vrai négatif"])].head(1)
        worst = df_polarite.loc[df_polarite["Véracité polarité"].isin(["✅ Vrai positif", "✅ Vrai négatif"])].tail(1)
        if not best.empty:
            st.success(f"🏅 Meilleur : **{best.iloc[0]['Sous-thème']}** ({best.iloc[0]['Polarité estimée']})")
        if not worst.empty:
            st.error(f"💔 Pire : **{worst.iloc[0]['Sous-thème']}** ({worst.iloc[0]['Polarité estimée']})")

    # === Exemples de verbatims ===
    if options["show_verbatims_examples"]:
        st.markdown("### 📝 Exemples de verbatims par sous-thème")
        for col in subtheme_cols[:5]:
            st.markdown(f"**{col}**")
            subset = df_enriched[df_enriched[col].notna()].head(3)
            for _, row in subset.iterrows():
                st.markdown(f"- {row['Verbatim public']} (Note IA {row['Note IA']})")

    # === Graphe corrélation × accélération ===
    if options["show_corr_accel"]:
        stats_bulles = utils.calculer_bulles_corr_accel(df_enriched, subtheme_cols, note_col="Note IA")
        fig = utils.tracer_bulles_corr_accel_plotly(stats_bulles)
        st.plotly_chart(fig, use_container_width=True)

    # === Rapport IA ===
    if st.button("📝 Générer un rapport IA"):
        generer_et_afficher_rapport(
            df_polarite,
            titre="Rapport de synthèse (Notes IA)",
            filename="rapport_ia_rating.pdf"
        )

    # === Export CSV final ===
    st.subheader("⬇️ Export des résultats")
    csv_bytes = utils.preparer_csv_export(df_enriched, "resultats_ia_rating.csv")
    st.download_button("Télécharger résultats", data=csv_bytes, file_name="resultats_ia_rating.csv", mime="text/csv")
