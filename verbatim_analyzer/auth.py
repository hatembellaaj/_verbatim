import json
import hashlib
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

USERS_FILE = Path(__file__).resolve().parent / "users.json"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _bootstrap_default_admin() -> None:
    """Ensure a default admin user exists so authentication is usable."""
    if USERS_FILE.exists():
        return

    default_users = {
        "admin": {"password": _hash_password("admin"), "role": "admin"}
    }
    save_users(default_users)


def load_users() -> Dict[str, Dict[str, str]]:
    _bootstrap_default_admin()
    try:
        with USERS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_users(users: Dict[str, Dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def verify_credentials(username: str, password: str) -> Optional[Dict[str, str]]:
    users = load_users()
    stored = users.get(username)
    if stored and stored.get("password") == _hash_password(password):
        return {"username": username, "role": stored.get("role", "user")}
    return None


def add_user(username: str, password: str, role: str) -> tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Identifiant et mot de passe sont requis."

    users = load_users()
    if username in users:
        return False, "L'utilisateur existe déjà."

    users[username] = {"password": _hash_password(password), "role": role}
    save_users(users)
    return True, f"Utilisateur {username} ajouté."


def change_password(username: str, current_password: str, new_password: str) -> tuple[bool, str]:
    """Update the password for a user after validating the current one."""

    if not new_password:
        return False, "Le nouveau mot de passe est requis."

    users = load_users()
    stored_user = users.get(username)

    if not stored_user:
        return False, "Utilisateur introuvable."

    if stored_user.get("password") != _hash_password(current_password):
        return False, "Le mot de passe actuel est incorrect."

    users[username]["password"] = _hash_password(new_password)
    save_users(users)
    return True, "Mot de passe mis à jour avec succès."


def require_authentication() -> Dict[str, str]:
    """Render a login form if necessary and return the authenticated user."""
    _bootstrap_default_admin()

    if "auth_user" in st.session_state:
        return st.session_state["auth_user"]

    with st.sidebar.form("login_form"):
        st.markdown("### 🔐 Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        user = verify_credentials(username, password)
        if user:
            st.session_state["auth_user"] = user
            st.success("Connexion réussie.")
            st.rerun()
        else:
            st.error("Identifiants invalides.")

    st.stop()


def render_user_badge(user: Dict[str, str]) -> None:
    st.sidebar.success(f"Connecté en tant que {user['username']} ({user['role']})")

    with st.sidebar.expander("🔑 Modifier mon mot de passe"):
        with st.form("change_password_form"):
            current_password = st.text_input("Mot de passe actuel", type="password")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le nouveau mot de passe", type="password")
            submitted = st.form_submit_button("Mettre à jour")

        if submitted:
            if new_password != confirm_password:
                st.sidebar.error("Les nouveaux mots de passe ne correspondent pas.")
            else:
                ok, message = change_password(user["username"], current_password, new_password)
                if ok:
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)

    if st.sidebar.button("Se déconnecter"):
        st.session_state.pop("auth_user", None)
        st.rerun()


def render_user_management(user: Dict[str, str]) -> None:
    if user.get("role") != "admin":
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Gestion des utilisateurs")
    with st.sidebar.form("add_user_form"):
        new_username = st.text_input("Nouvel utilisateur")
        new_password = st.text_input("Mot de passe", type="password")
        new_role = st.selectbox("Rôle", ["user", "admin"], index=0)
        submitted = st.form_submit_button("Ajouter l'utilisateur")

    if submitted:
        ok, message = add_user(new_username, new_password, new_role)
        if ok:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
