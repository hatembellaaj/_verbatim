import streamlit as st
import pandas as pd
from bertopic import BERTopic
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📚 Extraction automatique de thèmes avec BERTopic")

# Étape 1 : Upload CSV
file = st.file_uploader("📥 Chargez un fichier CSV avec une colonne texte (verbatims)", type="csv")

if file:
    df = pd.read_csv(file, sep=";", encoding="utf-8")
    colonnes_textuelles = [col for col in df.columns if df[col].dtype == "object"]
    colonne_texte = st.selectbox("📝 Choisissez la colonne contenant les verbatims :", colonnes_textuelles)

    if st.button("🧠 Extraire les thèmes avec BERTopic"):
        texts = df[colonne_texte].dropna().astype(str).tolist()
        
        with st.spinner("Chargement du modèle BERTopic..."):
            topic_model = BERTopic(language="multilingual", calculate_probabilities=True)
            topics, probs = topic_model.fit_transform(texts)
        
        df['theme_auto'] = topics

        # Affichage des thèmes principaux
        st.subheader("📌 Thèmes principaux détectés")
        freqs = topic_model.get_topic_info().head(10)
        st.dataframe(freqs)

        # Graphe interactif
        st.subheader("📊 Visualisation des thèmes")
        fig = topic_model.visualize_barchart(top_n_topics=10)
        st.plotly_chart(fig)

        # Export
        st.download_button("💾 Télécharger les résultats", df.to_csv(index=False).encode("utf-8"), "themes_bertopic.csv", "text/csv")
