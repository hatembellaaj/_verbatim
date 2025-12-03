from verbatim_analyzer.report_generator import generer_rapport_openai, exporter_rapport_pdf
import streamlit as st


def generer_et_afficher_rapport(df_stats, titre="Rapport de synthèse", filename="rapport.pdf"):
    """
    Génère un rapport avec OpenAI et propose un téléchargement en PDF.
    - df_stats : DataFrame contenant les stats (polarité, notes…)
    - titre : Titre affiché dans Streamlit
    - filename : nom du fichier PDF exporté
    """
    try:
        rapport = generer_rapport_openai(df_stats)
        if not rapport:
            st.warning("⚠️ Aucun contenu généré pour le rapport.")
            return

        st.markdown(f"### 📄 {titre}")
        st.markdown(rapport)

        pdf_bytes = exporter_rapport_pdf(rapport)
        st.download_button(
            "⬇️ Télécharger rapport PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport : {e}")
