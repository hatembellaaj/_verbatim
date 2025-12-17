import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _guess_default_column(columns: List[str], target_name: str) -> Optional[str]:
    """Return a column name matching target_name (case-insensitive) if it exists."""
    lower_map = {col.lower(): col for col in columns}
    return lower_map.get(target_name.lower())


def render_column_mapper(
    df: pd.DataFrame,
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
    key_prefix: str = "colmap",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Render Streamlit widgets to map required/optional fields to CSV columns.

    Returns a renamed dataframe (columns aligned with the expected field names)
    and the mapping chosen by the user. A stop is triggered if the mapping is
    invalid (missing required fields or duplicated selections).
    """

    if df.empty or not list(df.columns):
        st.error("❌ Le fichier ne contient aucune colonne exploitable.")
        st.stop()

    st.subheader("🗂️ Associer les colonnes du fichier")
    st.caption(
        "Sélectionnez pour chaque champ requis la colonne correspondante dans votre fichier CSV."
    )

    mapping: Dict[str, str] = {}
    columns = list(df.columns)

    # Gestion des champs obligatoires
    for field in required_fields:
        default = _guess_default_column(columns, field)
        selected = st.selectbox(
            f"Colonne pour « {field} »",
            options=columns,
            index=columns.index(default) if default is not None else 0,
            key=f"{key_prefix}_req_{field}",
        )
        mapping[field] = selected

    # Gestion des champs optionnels
    for field in optional_fields or []:
        choices = ["— Aucune —"] + columns
        default = _guess_default_column(columns, field)
        default_index = choices.index(default) if default is not None else 0
        selected = st.selectbox(
            f"Colonne pour « {field} » (optionnel)",
            options=choices,
            index=default_index,
            key=f"{key_prefix}_opt_{field}",
        )
        if selected != "— Aucune —":
            mapping[field] = selected

    # Validation : aucune colonne ne doit être utilisée plusieurs fois
    selected_columns = list(mapping.values())
    if len(selected_columns) != len(set(selected_columns)):
        st.error("❌ Chaque colonne du fichier ne peut être utilisée qu'une seule fois.")
        st.stop()

    renamed_df = df.rename(columns={value: key for key, value in mapping.items()})
    missing_after_rename = [field for field in required_fields if field not in renamed_df.columns]
    if missing_after_rename:
        st.error(
            "❌ Mapping incomplet : veuillez sélectionner une colonne pour "
            + ", ".join(missing_after_rename)
        )
        st.stop()

    return renamed_df, mapping


def load_csv_with_mapping(
    uploaded_file,
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
    key_prefix: str = "colmap",
    sample_rows: int = 200,
) -> pd.DataFrame:
    """Charge un CSV en deux temps pour fiabiliser le mapping des colonnes.

    1) Lecture d'un échantillon pour détecter le séparateur et proposer le mapping.
    2) Relecture complète en ne conservant que les colonnes mappées (usecols).
    """

    try:
        uploaded_file.seek(0)
        preview = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            nrows=sample_rows,
            on_bad_lines="skip",
        )
    except Exception as e:
        st.error(f"Erreur lors de la lecture d'un échantillon du fichier : {e}")
        st.stop()

    preview, mapping = render_column_mapper(
        preview,
        required_fields=required_fields,
        optional_fields=optional_fields,
        key_prefix=key_prefix,
    )

    usecols = list(mapping.values())

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            usecols=usecols,
            on_bad_lines="skip",
        )
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier complet après mapping : {e}")
        st.stop()

    renamed_df = df.rename(columns={value: key for key, value in mapping.items()})
    return renamed_df
