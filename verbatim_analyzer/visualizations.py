import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def plot_granularity(df_result):
    if "x" in df_result.columns and "y" in df_result.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_result,
            x="x", y="y",
            hue="Cluster",
            palette="tab10",
            s=80
        )
        plt.title("Projection des verbatims (réduction de dimension)")
        st.pyplot(fig)
    else:
        st.warning("Les colonnes 'x' et 'y' sont absentes. Pas de projection possible.")

def plot_profile_theme_matrix(df_result):
    if "Cluster" in df_result.columns and "Êtes-vous venu :" in df_result.columns:
        matrix = pd.crosstab(df_result["Êtes-vous venu :"], df_result["Cluster"])
        st.dataframe(matrix)
    else:
        st.warning("Impossible de générer la matrice profil/thèmes. Colonnes manquantes.")

def plot_cluster_pie_chart(df_result):
    if "Theme" in df_result.columns:
        counts = df_result["Theme"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        plt.title("Répartition des thèmes")
        st.pyplot(fig)
    else:
        st.warning("Pas de colonne 'Theme' pour générer le camembert.")

def plot_profile_pie_chart(df_result):
    if "Profil" in df_result.columns:
        counts = df_result["Profil"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        plt.title("Répartition des profils")
        st.pyplot(fig)
    else:
        st.warning("Pas de colonne 'Profil' pour générer le camembert.")



def plot_profile_theme_overrepresentation(df_result):
    """
    Affiche une heatmap de sur-/sous-représentation des thèmes par rapport aux profils.
    """
    if "Profil" not in df_result.columns or "Cluster" not in df_result.columns:
        st.warning("Colonnes manquantes pour calculer la sur-représentation.")
        return

    # Calcul de la distribution des profils dans l'ensemble des données
    total_profile_dist = df_result["Profil"].value_counts(normalize=True)

    # Distribution conditionnelle des profils par cluster
    cluster_profile_dist = df_result.groupby("Cluster")["Profil"].value_counts(normalize=True).unstack().fillna(0)

    # Sur-/sous-représentation = différence entre la distribution conditionnelle et globale
    overrep_matrix = cluster_profile_dist.subtract(total_profile_dist, axis=1).round(2)

    st.subheader("🔥 Sur-/Sous-représentation des profils dans chaque thème")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(overrep_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
