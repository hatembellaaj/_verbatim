# analyze_combined.py
import streamlit as st
import pandas as pd
import plotly.express as px
import utils
from sidebar_options import get_sidebar_options
from report_utils import generer_et_afficher_rapport
from verbatim_analyzer.marketing_analyzer import extract_marketing_clusters_with_openai, associer_sous_themes_par_similarity
from streamlit_tree_select import tree_select

def run():
    st.title("🧩 Analyse complète des verbatims")
    st.markdown("""
    Bienvenue dans le module **Analyse combinée**.  
    Suivez les étapes ci-dessous pour explorer vos verbatims selon les approches **Marketing** ou **IA Rating**.
    """)

    # === Étape 1 : Chargement fichier ===
    st.header("📂 Étape 1 : Import du fichier")
    uploaded_file = st.file_uploader("Téléversez un fichier CSV", type="csv")

    if uploaded_file is None:
        st.info("En attente d’un fichier CSV…")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ {len(df)} lignes chargées")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # Validation colonnes
    if "Verbatim public" not in df.columns:
        st.error("❌ Le fichier doit contenir la colonne 'Verbatim public'.")
        st.stop()

    df["Verbatim complet"] = df["Verbatim public"].fillna("") + " " + df.get("Verbatim privé", "").fillna("")

    # === Étape 2 : Choix du mode d’analyse ===
    st.header("⚙️ Étape 2 : Choisissez le mode d’analyse")
    mode = st.radio(
        "Quelle approche souhaitez-vous utiliser ?",
        ["Analyse Marketing (note client)", "Analyse IA (note générée automatiquement)"],
        horizontal=True
    )

    options = get_sidebar_options(uploaded_file)

    # === Étape 3 : Paramétrage & Extraction des thèmes ===
    if st.button("🔄 Ré-extraire les clusters via OpenAI"):
        if "themes_extraits" in st.session_state:
            del st.session_state["themes_extraits"]
        st.rerun()
    
    st.header("🧠 Étape 3 : Définition des thèmes")
    col1, col2 = st.columns(2)

    with col1:
        use_openai = st.toggle("Utiliser OpenAI pour extraire les clusters", options["use_openai"])
    with col2:
        nb_clusters = st.slider("Nombre de clusters (si OpenAI)", 3, 15, options["nb_clusters"])

    themes = []

    # 🔁 Si on a déjà des thèmes extraits en mémoire, on les réutilise
    if "themes_extraits" in st.session_state:
        themes = st.session_state["themes_extraits"]

    # ⚙️ Extraction seulement si OpenAI activé ET pas déjà fait
    elif use_openai:
        with st.spinner("Extraction automatique via OpenAI..."):
            try:
                texts_public = df["Verbatim public"].astype(str).tolist()
                texts_private = df["Verbatim privé"].astype(str).tolist() if "Verbatim privé" in df.columns else [""] * len(df)
                themes = extract_marketing_clusters_with_openai(texts_public, texts_private, nb_clusters)
                st.session_state["themes_extraits"] = themes
                st.success("✅ Clusters extraits automatiquement")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur OpenAI : {e}")
                st.stop()

    else:
        user_themes = st.text_area("Thèmes manuels (JSON ou CSV)").strip()
        if not user_themes:
            st.warning("⚠️ Fournissez une liste de thèmes manuelle ou activez OpenAI.")
            st.stop()
        try:
            themes = pd.read_json(user_themes).to_dict(orient="records")
        except Exception:
            themes = [{"theme": t.strip(), "subthemes": []} for t in user_themes.split(",") if t.strip()]
        st.success(f"✅ {len(themes)} thèmes chargés manuellement")


    # --- Modification / Ajout de clusters ---
    st.divider()
    st.markdown("### ✏️ Modification / Ajout de clusters")

    # Ajouter un nouveau thème
    with st.expander("➕ Ajouter un nouveau thème"):
        new_theme = st.text_input("Nom du nouveau thème")
        if st.button("Ajouter le thème"):
            if new_theme and all(t["theme"] != new_theme for t in themes):
                themes.append({"theme": new_theme, "subthemes": []})
                st.session_state["themes_extraits"] = themes
                st.success(f"Thème **{new_theme}** ajouté ✅")
                st.rerun()

    # Ajouter un sous-thème à un thème existant
    with st.expander("➕ Ajouter un sous-thème à un thème existant"):
        theme_choice = st.selectbox("Sélectionnez le thème parent", [t["theme"] for t in themes])
        new_sub = st.text_input("Nom du nouveau sous-thème")
        new_keywords = st.text_input("Mots-clés associés (séparés par une virgule)")
        if st.button("Ajouter le sous-thème"):
            if new_sub:
                keywords_list = [kw.strip() for kw in new_keywords.split(",") if kw.strip()]
                for t in themes:
                    if t["theme"] == theme_choice:
                        t.setdefault("subthemes", []).append({"label": new_sub, "keywords": keywords_list})
                        break
                st.session_state["themes_extraits"] = themes
                st.success(f"Sous-thème **{new_sub}** ajouté à **{theme_choice}** ✅")
                st.rerun()

    # --- Arborescence interactive ---
    st.divider()
    st.markdown("### 🌳 Arborescence des clusters détectés / définis")

    def convertir_en_tree_data(themes):
        data = []
        for t in themes:
            children = []
            for s in t.get("subthemes", []):
                label = s.get("label") if isinstance(s, dict) else s
                keywords = s.get("keywords", []) if isinstance(s, dict) else []
                keyword_hint = f" — mots-clés: {', '.join(keywords)}" if keywords else ""
                children.append({
                    "label": f"{label}{keyword_hint}",
                    "value": f"{t['theme']}::{label}"
                })
            data.append({
                "label": t.get("theme", "Thème sans nom"),
                "value": t.get("theme", "Thème sans nom"),
                "children": children
            })
        return data

    tree_data = convertir_en_tree_data(themes)

    selected_nodes = tree_select(
        tree_data,
        "Sélectionnez les thèmes et sous-thèmes à retenir",
        key="cluster_tree"
    )

    if selected_nodes and selected_nodes.get("checked"):
        st.session_state["selected_clusters"] = selected_nodes["checked"]
        st.success(f"📂 Clusters validés : {', '.join(st.session_state['selected_clusters'])}")
    else:
        st.info("🟡 Aucun cluster validé dans l’arbre.")

    def filtrer_themes(themes, selection):
        selection_set = set(selection or [])
        filtres = []
        for t in themes:
            theme_name = t.get("theme", "")
            subthemes = []
            for s in t.get("subthemes", []):
                label = s.get("label") if isinstance(s, dict) else s
                value = f"{theme_name}::{label}" if label else theme_name
                if value in selection_set:
                    subthemes.append(s)
            if theme_name in selection_set:
                filtres.append(t)
            elif subthemes:
                filtres.append({"theme": theme_name, "subthemes": subthemes})
        return filtres

    themes_selectionnes = filtrer_themes(themes, st.session_state.get("selected_clusters", []))

    # Aperçu du JSON final réellement utilisé
    with st.expander("📜 JSON final des thèmes sélectionnés"):
        if themes_selectionnes:
            st.json(themes_selectionnes)
        else:
            st.info("Aucun cluster sélectionné pour le moment.")

    # 🚦 Blocage tant que rien n’est sélectionné
    if "selected_clusters" not in st.session_state or not st.session_state["selected_clusters"]:
        st.warning("⚠️ Vous devez valider au moins un cluster avant de continuer.")
        st.stop()



    # === Étape 4 : Calcul des notes ===
    st.header("💬 Étape 4 : Calcul des notes")
    if mode.startswith("Analyse IA"):
        st.info("Les notes sont générées automatiquement par IA (1 à 5)")
        df["Note IA"] = df["Verbatim public"].astype(str).apply(utils.calculer_note_ia)
        note_col = "Note IA"
    else:
        if "Note globale avis 1" not in df.columns:
            st.error("❌ Le fichier doit contenir une colonne 'Note globale avis 1'.")
            st.stop()
        note_col = "Note globale avis 1"

    st.success(f"✅ Notes prêtes ({note_col})")

    # === Étape 5 : Association sous-thèmes ===
    st.write("DEBUG options =", options)

    # Gestion de plusieurs formats possibles de retour
    if isinstance(options, dict):
        model_choice = options.get("model_choice", "MiniLM")
        seuil_similarite = options.get("seuil_similarite", 0.75)
    elif isinstance(options, list):
        # Si c’est une liste de dictionnaires
        if len(options) > 0 and isinstance(options[0], dict):
            model_choice = options[0].get("model_choice", "MiniLM")
            seuil_similarite = options[0].get("seuil_similarite", 0.75)
        else:
            # Valeurs par défaut si c’est une liste simple
            model_choice, seuil_similarite = "MiniLM", 0.75
    else:
        model_choice, seuil_similarite = "MiniLM", 0.75

    # Choix du modèle selon l’option
    model_name = "all-MiniLM-L6-v2" if model_choice == "MiniLM" else "bert-base-nli-mean-tokens"

    # Utiliser uniquement les clusters validés pour la suite
    themes_utilises = themes_selectionnes if themes_selectionnes else themes
    st.session_state["themes_valides"] = themes_utilises

    # Association des sous-thèmes
    df_enriched = associer_sous_themes_par_similarity(
        df,
        themes=themes_utilises,
        text_col="Verbatim complet",
        model_name=model_name,
        seuil_similarite=seuil_similarite
    )
    subtheme_cols = [c for c in df_enriched.columns if "::" in c]
    if not subtheme_cols:
        st.warning("⚠️ Aucun sous-thème détecté.")
        st.stop()

    # === Étape 6 bis : Vérification des incohérences ===
    st.header("🧩 Étape 6 bis : Vérification des incohérences sémantiques")

    if st.toggle("Activer la détection des incohérences", value=False):
        with st.spinner("🔍 Vérification des incohérences en cours..."):
            incoherences = utils.verifier_coherence_semantique(
                df_enriched, subtheme_cols, seuil=0.3, alpha=0.7
            )

        if incoherences.empty:
            st.success("✅ Aucune incohérence détectée.")
        else:
            st.warning(f"⚠️ {len(incoherences)} incohérences détectées.")
            st.dataframe(incoherences)

            # Construction de la matrice finale
            matrice_finale = utils.construire_matrice_finale(df_enriched, incoherences, subtheme_cols)
            st.markdown("### 📐 Matrice finale (avec décisions)")
            st.dataframe(matrice_finale)

            # Suppression des incohérences dans df_enriched
            avant = len(df_enriched)
            for _, row in incoherences.iterrows():
                verb = row["Verbatim"]
                sous_theme = row["Sous-thème"]
                if row["Decision"] in ["⚠️ Suspect", "Non retenue"]:
                    df_enriched.loc[df_enriched["Verbatim complet"] == verb, sous_theme] = pd.NA
            df_enriched = df_enriched[df_enriched[subtheme_cols].notna().any(axis=1)]
            apres = len(df_enriched)

            st.info(f"🧹 Nettoyage : {avant - apres} verbatims incohérents supprimés ({apres} restants).")

            # Export optionnel
            csv_incoh = utils.preparer_csv_export(incoherences, "incoherences_detectees.csv")
            st.download_button(
                "⬇️ Télécharger les incohérences",
                data=csv_incoh,
                file_name="incoherences_detectees.csv",
                mime="text/csv"
            )

        st.success("✅ Nettoyage terminé, passage à la visualisation possible.")



    # === Étape 6 : Visualisations ===
    st.header("📊 Étape 6 : Visualisation des résultats")
    tabs = st.tabs(["📈 Statistiques", "💬 Verbatims", "🥧 Répartition", "🧾 Rapport"])

    with tabs[0]:
        st.subheader("Moyenne des notes par sous-thème")
        df_melted = df_enriched[[note_col] + subtheme_cols].melt(id_vars=note_col, var_name="Sous-thème", value_name="Assoc").dropna()
        stats = df_melted.groupby("Sous-thème")[note_col].agg(["count", "mean"]).sort_values("mean", ascending=False)
        st.dataframe(stats)
        fig = px.bar(stats, x=stats.index, y="mean", title="Moyenne des notes par sous-thème")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Exemples de verbatims")
        for col in subtheme_cols[:5]:
            subset = df_enriched[df_enriched[col].notna()].head(3)
            st.markdown(f"**{col}**")
            for _, row in subset.iterrows():
                st.markdown(f"- {row['Verbatim public']} ({note_col}: {row[note_col]})")

    with tabs[2]:
        st.subheader("Répartition des sous-thèmes")
        counts = df_enriched[subtheme_cols].notna().sum().reset_index()
        counts.columns = ["Sous-thème", "Occurrences"]
        fig_pie = px.pie(counts, names="Sous-thème", values="Occurrences", title="Répartition des verbatims par sous-thème")
        st.plotly_chart(fig_pie, use_container_width=True)

    with tabs[3]:
        if st.button("📝 Générer le rapport complet"):
            generer_et_afficher_rapport(
                stats,
                titre=f"Rapport synthèse - {mode}",
                filename=f"rapport_{'ia' if 'IA' in mode else 'marketing'}.pdf"
            )

    # === Étape 7 : Export CSV ===
    st.header("⬇️ Étape 7 : Export des résultats")
    csv_bytes = utils.preparer_csv_export(df_enriched, f"resultats_{'ia' if 'IA' in mode else 'marketing'}_fusion.csv")
    st.download_button("Télécharger les résultats", data=csv_bytes, file_name="resultats_combined.csv", mime="text/csv")
