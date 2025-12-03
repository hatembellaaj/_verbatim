import streamlit as st
import marketing
import ia_rating  # nouveau nom du module
import analyze_combined

st.set_page_config(layout="wide")
st.title("🧠 Analyse des verbatims client")

menu = st.sidebar.selectbox("Navigation", ["Marketing", "IA Rating", "Analyse combinée"])

if menu == "Marketing":
    marketing.run()
elif menu == "IA Rating":
    ia_rating.run()
elif menu == "Analyse combinée":
    analyze_combined.run()
