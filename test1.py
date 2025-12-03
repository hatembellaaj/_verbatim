import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
from keybert import KeyBERT
from transformers import pipeline
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


st.set_page_config(layout="wide")
st.title("🧠 Analyse d'avis clients avec détection automatique + KeyBERT")

# Upload du fichier
file = st.file_uploader("📁 Téléversez un fichier CSV (avis clients)", type="csv")

if file:
    # Chargement des données
    df = pd.read_csv(file, sep=';', encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
    df.columns = ['date_avis', 'date_fin_experience', 'profil_client', 'verbatim_public', 'verbatim_prive']
    df['texte_complet'] = (df['verbatim_public'].fillna('') + ' ' + df['verbatim_prive'].fillna('')).str.lower()
    df = df[df['texte_complet'].str.strip() != '']
    st.success(f"{len(df)} avis chargés.")

    # Analyse de sentiment avec BERT
    with st.spinner("🔍 Analyse de sentiment avec BERT..."):
        analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        df['sentiment'] = df['texte_complet'].apply(lambda x: int(analyzer(x[:512])[0]['label'][0]))

    # Extraction de mots-clés avec KeyBERT
    if st.checkbox("🧠 Extraire automatiquement les mots-clés avec KeyBERT"):
        with st.spinner("Chargement de KeyBERT..."):
            kw_model = KeyBERT(model='all-MiniLM-L6-v2')

        ngram_list = []
        for i, texte in enumerate(df['texte_complet'].head(50)):
            if texte.strip():
                keywords = kw_model.extract_keywords(texte, keyphrase_ngram_range=(1, 3), stop_words='french', top_n=3)
                kw_clean = [kw[0] for kw in keywords]
                df.at[i, 'keybert_keywords'] = ", ".join(kw_clean)
                ngram_list.extend(kw_clean)

        # Affichage
        st.subheader("🗝️ Mots-clés extraits")
        st.dataframe(df[['texte_complet', 'keybert_keywords']].head(10))

        # Fréquences
        freq = Counter(ngram_list).most_common(10)
        df_keywords = pd.DataFrame(freq, columns=["mot-clé", "fréquence"]).set_index("mot-clé")
        st.bar_chart(df_keywords)

    # Graphique des sentiments
    st.subheader("📊 Répartition des notes de sentiment")
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment'], bins=5, kde=False, ax=ax)
    ax.set_title("Distribution des scores de sentiment")
    ax.set_xlabel("Score (1 à 5)")
    st.pyplot(fig)

    # Export
    st.download_button("📥 Télécharger les résultats", df.to_csv(index=False).encode('utf-8'), "resultats.csv", "text/csv")

if st.checkbox("🧠 Regrouper automatiquement les clients (clustering KMeans)"):
    st.subheader("📌 Profils types détectés automatiquement")

    # Étape 1 : vectorisation des textes

    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    X = vectorizer.fit_transform(df['texte_complet'])



    # Étape 2 : clustering (ex: 5 clusters)
    n_clusters = st.slider("Nombre de profils à détecter", 2, 10, 5)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['persona_auto'] = model.fit_predict(X)

    # Affichage des clusters
    st.write(df[['texte_complet', 'persona_auto']].head(10))

    # Statistiques par cluster
    st.subheader("📈 Sentiment moyen par persona détecté")
    stats = df.groupby("persona_auto")["sentiment"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=stats, x="persona_auto", y="sentiment", ax=ax)
    ax.set_title("Sentiment moyen par persona détecté")
    st.pyplot(fig)