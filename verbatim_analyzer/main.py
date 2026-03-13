import logging
import streamlit as st
import marketing
import ia_rating  # nouveau nom du module
import analyze_combined
import manual
import auth
import os

logging.basicConfig(
    level=os.getenv("APP_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# streamlit-tree-select references a missing source map file (bootstrap.min.css.map)
# in its packaged frontend assets. Streamlit logs this as an internal component
# request traceback even though the app continues to work normally.
logging.getLogger("streamlit.web.server.component_request_handler").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")
logger.info("Application Streamlit initialisée")

user = auth.require_authentication()

header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("🧠 Analyse des verbatims client")
with header_right:
    top_menu = st.selectbox(
        "Navigation générale",
        ["Application", "Manuel d'utilisation"],
        index=0,
    )

auth.render_user_badge(user)
auth.render_user_management(user)

if top_menu == "Manuel d'utilisation":
    manual.render_manual()
    st.stop()

menu_options = ["Marketing", "IA Rating", "Analyse combinée"]
menu = st.sidebar.selectbox("Navigation", menu_options, index=menu_options.index("Analyse combinée"))

if menu == "Marketing":
    marketing.run()
elif menu == "IA Rating":
    ia_rating.run()
elif menu == "Analyse combinée":
    analyze_combined.run()
