"""
Explorateur de Recherche Textuelle - Application Streamlit Éducative
Application pédagogique pour enseigner TF-IDF, BM25, et autres techniques de recherche
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tfidf_engine import TFIDFEngine
from bm25_engine import BM25Engine
from data_loader import load_dataset

# Imports optionnels pour Embeddings (nécessite sentence-transformers)
try:
    EMBEDDINGS_AVAILABLE = True

    # Import des sections Embeddings et Synthèse (si dépendances disponibles)
    from app_embeddings_sections import (
        render_embeddings_section,
    )
    from app_synthesis_sections import render_synthesis_section
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Le warning sera affiché dans la sidebar

# Import des sections TF-IDF et BM25 (toujours disponibles)
from app_tfidf_sections import (
    render_tfidf_section,
)
from app_bm25_sections import (
    render_bm25_section,
)


# Configuration de la page
st.set_page_config(
    page_title="Explorateur de Recherche Textuelle 🔍",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Style CSS personnalisé
st.markdown(
    """
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# FONCTIONS DE CACHE
# ============================================================================


@st.cache_data(show_spinner="Chargement du dataset...")
def load_cached_dataset(
    dataset_name: str,
    sample_size: int = None,
    extended: bool = False,
    _version: int = 3,
):
    """Charge un dataset avec cache (version 3 - tailles corrigées)"""
    return load_dataset(dataset_name, sample_size=sample_size, extended=extended)


@st.cache_resource
def create_tfidf_engine(documents_texts: list, remove_stopwords: bool = True):
    """Crée et entraîne le moteur TF-IDF avec cache"""
    engine = TFIDFEngine(documents_texts, remove_stopwords=remove_stopwords)
    engine.fit()
    return engine


@st.cache_resource
def create_bm25_engine(
    documents_texts: list,
    k1: float = 1.5,
    b: float = 0.75,
    remove_stopwords: bool = True,
):
    """Crée le moteur BM25 avec cache"""
    return BM25Engine(documents_texts, k1=k1, b=b, remove_stopwords=remove_stopwords)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def render_tab_navigation(
    tabs_list: list, session_key: str, default_tab: str = None
) -> str:
    """
    Rend une navigation par tabs avec des boutons stylés

    Args:
        tabs_list: Liste des noms de tabs
        session_key: Clé pour le session_state
        default_tab: Tab par défaut (premier si None)

    Returns:
        Le tab actuellement sélectionné
    """
    # Initialiser le state si nécessaire
    if session_key not in st.session_state:
        st.session_state[session_key] = default_tab or tabs_list[0]

    # Créer des colonnes pour les boutons horizontaux
    cols = st.columns(len(tabs_list))

    for idx, (col, tab_name) in enumerate(zip(cols, tabs_list)):
        with col:
            if st.session_state[session_key] == tab_name:
                # Tab actif - afficher avec style
                st.markdown(
                    f"""
                <div style="
                    background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                    padding: 10px 5px;
                    border-radius: 6px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    font-size: 0.9rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                    margin-bottom: 10px;
                ">
                    {tab_name}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Bouton cliquable
                if st.button(
                    tab_name, key=f"{session_key}_{idx}", use_container_width=True
                ):
                    st.session_state[session_key] = tab_name
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    return st.session_state[session_key]


def get_example_queries(dataset_name: str) -> dict:
    """
    Retourne des exemples de queries et placeholders pour chaque dataset

    Args:
        dataset_name: Nom du dataset ('recettes', 'films', 'wikipedia')

    Returns:
        Dict avec 'placeholder' et 'queries' (liste d'exemples)
    """
    examples = {
        "recettes": {
            "placeholder": "Ex: plat italien, cuisine épicée, dessert chocolat...",
            "queries": [
                "plat italien pâtes fromage",
                "cuisine asiatique épicée crevettes",
                "dessert chocolat français",
                "poisson grillé méditerranéen",
            ],
        },
        "films": {
            "placeholder": "Ex: science-fiction espace, comédie romantique...",
            "queries": [
                "science-fiction espace vaisseau",
                "comédie romantique amour couple",
                "super-héros action marvel",
                "film horreur suspense peur",
            ],
        },
        "wikipedia": {
            "placeholder": "Ex: guerre mondiale, intelligence artificielle...",
            "queries": [
                "guerre mondiale conflit armée",
                "intelligence artificielle machine learning",
                "football coupe monde champion",
                "physique quantique atome particule",
            ],
        },
    }
    return examples.get(dataset_name, examples["recettes"])


# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================


def render_datasets_section(dataset_name: str, use_extended: bool):
    """Section d'exploration des datasets (DÉBOGAGE!)"""
    st.markdown(
        '<h1 class="main-title">📦 Explorateur de Datasets</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Explore et vérifie les données chargées!</p>',
        unsafe_allow_html=True,
    )

    # Charger le dataset
    with st.spinner("🔄 Chargement du dataset..."):
        dataset = load_cached_dataset(dataset_name, extended=use_extended)

    # === INFOS DATASET ===
    st.header("📊 Informations du Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Nombre de documents", len(dataset))
    with col2:
        categories = list(set(doc.get("category", "N/A") for doc in dataset))
        st.metric("🏷️ Catégories", len(categories))
    with col3:
        avg_length = np.mean([len(doc["text"].split()) for doc in dataset])
        st.metric("📝 Longueur moyenne", f"{avg_length:.0f} mots")
    with col4:
        total_words = sum([len(doc["text"].split()) for doc in dataset])
        st.metric("💬 Total de mots", f"{total_words:,}")

    # Source des données
    st.markdown("---")
    st.subheader("🔍 Source des Données")

    # Récupérer les infos du dataset
    from src.data_loader import get_dataset_info

    dataset_info = get_dataset_info(dataset_name)

    st.success(f"✅ **Chargé depuis:** {dataset_info['source']}")

    st.info(f"""
    **Dataset actuel:** `{dataset_name}`
    **Taille:** `{"Extended" if use_extended else "Normal"}`
    **Documents chargés:** `{len(dataset)}`
    **Fichier:** `{dataset_info["file"]}`
    """)

    st.markdown("---")

    # === LISTE DES DOCUMENTS (PAGINÉE) ===
    st.header("📋 Liste des Documents")

    # Recherche/Filtrage
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_text = st.text_input(
            "🔎 Rechercher dans les titres:",
            placeholder="Ex: pizza, science-fiction, guerre...",
            help="Filtre les documents dont le titre contient ce texte",
        )
    with col2:
        selected_category = st.selectbox(
            "🏷️ Catégorie:", ["Toutes"] + sorted(categories)
        )
    with col3:
        page_size = st.selectbox(
            "📄 Par page:",
            options=[10, 25, 50, 100],
            index=1,
            help="Nombre de documents par page",
        )

    # Filtrer les documents
    filtered_docs = dataset
    if search_text:
        filtered_docs = [
            doc for doc in filtered_docs if search_text.lower() in doc["title"].lower()
        ]
    if selected_category != "Toutes":
        filtered_docs = [
            doc
            for doc in filtered_docs
            if doc.get("category", "N/A") == selected_category
        ]

    # Initialiser la pagination
    if "dataset_page" not in st.session_state:
        st.session_state.dataset_page = 0

    total_docs = len(filtered_docs)
    total_pages = max(1, (total_docs + page_size - 1) // page_size)

    # S'assurer que la page est valide
    if st.session_state.dataset_page >= total_pages:
        st.session_state.dataset_page = max(0, total_pages - 1)

    # Calculer les indices de la page
    start_idx = st.session_state.dataset_page * page_size
    end_idx = min(start_idx + page_size, total_docs)
    current_page_docs = filtered_docs[start_idx:end_idx]

    # Afficher les infos de pagination
    st.caption(
        f"📊 Affichage {start_idx + 1}-{end_idx} sur {total_docs} documents • Page {st.session_state.dataset_page + 1}/{total_pages}"
    )

    if total_docs == 0:
        st.warning("😕 Aucun document trouvé avec ces filtres!")
    else:
        # Créer le DataFrame pour affichage
        table_data = []
        for i, doc in enumerate(current_page_docs):
            table_data.append(
                {
                    "#": start_idx + i + 1,
                    "Titre": doc["title"][:80]
                    + ("..." if len(doc["title"]) > 80 else ""),
                    "Catégorie": doc.get("category", "N/A"),
                    "Mots": len(doc["text"].split()),
                    "_full_doc": doc,  # Stocker le doc complet (caché)
                }
            )

        df = pd.DataFrame(table_data)

        # Afficher le tableau interactif (sans la colonne _full_doc)
        df_display = df.drop(columns=["_full_doc"])

        st.info("💡 **Clique sur une ligne du tableau pour voir les détails!**", icon="💡")

        # Tableau interactif avec sélection
        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
            key=f"dataset_table_page_{st.session_state.dataset_page}"
        )

        # Contrôles de pagination
        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button(
                "⬅️ Précédent",
                disabled=(st.session_state.dataset_page == 0),
                use_container_width=True,
            ):
                st.session_state.dataset_page -= 1
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align: center; padding-top: 8px;'>Page {st.session_state.dataset_page + 1} / {total_pages}</div>",
                unsafe_allow_html=True,
            )

        with col_next:
            if st.button(
                "Suivant ➡️",
                disabled=(st.session_state.dataset_page >= total_pages - 1),
                use_container_width=True,
            ):
                st.session_state.dataset_page += 1
                st.rerun()

        st.markdown("---")

        # Détection de la sélection via le clic sur une ligne
        selected_doc_idx = None
        if event.selection and "rows" in event.selection and len(event.selection["rows"]) > 0:
            selected_doc_idx = event.selection["rows"][0]

        # Afficher les détails si une ligne est sélectionnée
        if selected_doc_idx is not None:
            doc = current_page_docs[selected_doc_idx]

            st.markdown("---")
            st.subheader("📄 Détails du Document")

            # Infos dans des colonnes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Titre", "")
                st.write(f"**{doc['title']}**")
            with col2:
                st.metric("🏷️ Catégorie", doc.get("category", "N/A"))
            with col3:
                word_count = len(doc["text"].split())
                st.metric("📊 Longueur", f"{word_count} mots")

            # Contenu complet
            st.markdown("**📖 Contenu complet:**")
            st.text_area(
                "Texte du document",
                value=doc["text"],
                height=300,
                label_visibility="collapsed",
            )

            # Statistiques du texte
            with st.expander("📊 Statistiques détaillées"):
                tokens = doc["text"].lower().split()
                unique_tokens = set(tokens)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mots totaux", len(tokens))
                with col2:
                    st.metric("Mots uniques", len(unique_tokens))
                with col3:
                    diversity = (
                        len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
                    )
                    st.metric("Diversité lexicale", f"{diversity:.2%}")

                # Mots les plus fréquents
                from collections import Counter

                word_freq = Counter(tokens)
                most_common = word_freq.most_common(10)

                st.markdown("**🔤 Top 10 mots les plus fréquents:**")
                df = pd.DataFrame(most_common, columns=["Mot", "Fréquence"])
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_home():
    """Page d'accueil avec présentation générale"""
    st.markdown(
        '<h1 class="main-title">🔍 Explorateur de Recherche Textuelle</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Maîtrise les techniques de recherche textuelle de A à Z!</p>',
        unsafe_allow_html=True,
    )

    # === INTRO COURTE ===
    st.info("""
    **🎯 Mission:** Apprendre comment les moteurs de recherche trouvent les meilleurs résultats parmi des milliers de documents.

    **Exemple:** Tu cherches _"dessert au chocolat"_ → Comment l'algorithme classe-t-il 10,000 recettes? 🍰
    """)

    st.markdown("---")

    # === PARCOURS (SIMPLIFIÉ) ===
    st.markdown("## 📚 Parcours d'Apprentissage")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 TF-IDF
        **Les Fondamentaux**

        🟢 Débutant • 15 min

        La technique **classique** pour pondérer l'importance des mots.

        ✅ Fréquences normalisées
        ✅ Mots rares = plus importants
        ✅ Similarité cosinus
        """)

    with col2:
        st.markdown("""
        ### 🎯 BM25
        **L'Amélioration**

        🟡 Intermédiaire • 20 min

        Version **améliorée** utilisée par les moteurs pro.

        ✅ Saturation intelligente
        ✅ Paramètres ajustables
        ✅ Meilleurs résultats
        """)

    with col3:
        if EMBEDDINGS_AVAILABLE:
            st.markdown("""
            ### 🧠 Embeddings
            **IA & Sémantique**

            🔴 Avancé • 30 min

            Recherche **moderne** par réseaux de neurones.

            ✅ Comprend le sens
            ✅ Trouve des synonymes
            ✅ Hybrid search
            """)
        else:
            st.markdown("""
            ### 🧠 Embeddings 🔒
            **IA & Sémantique**

            🔴 Avancé

            Installe les dépendances:
            ```bash
            pip install sentence-transformers
            ```
            """)

    st.success("💡 **Recommandé:** Suis l'ordre TF-IDF → BM25 → Embeddings → Synthèse")

    st.markdown("---")

    # === DATASETS (SIMPLIFIÉ + À JOUR) ===
    st.markdown("## 📦 Datasets Disponibles")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ### 🍝 Recettes
        **50 → 200 docs**

        Cuisine française, italienne, asiatique, mexicaine
        """)

    with col2:
        st.markdown("""
        ### 🎬 Films
        **50 → 200 docs**

        Synopsis de films variés (action, comédie, SF)
        """)

    with col3:
        st.markdown("""
        ### 📖 Livres
        **100 → 801 docs**

        Résumés de livres français (classiques & modernes)
        """)

    with col4:
        st.markdown("""
        ### 📚 Wikipedia
        **100 → 1K docs**

        Articles FR sur tech, histoire, science, sport
        """)

    st.markdown("---")

    # === GUIDE RAPIDE ===
    st.markdown("## 🚀 Démarrage Rapide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📍 Navigation**
        - Sidebar (←) → Choix section & dataset
        - Onglets → Intro, Concepts, Recherche, etc.
        - Benchmarks → Compare les performances
        """)

    with col2:
        st.markdown("""
        **🎓 Parcours Complet**
        1. 📊 TF-IDF (15 min)
        2. 🎯 BM25 (20 min)
        3. 🧠 Embeddings (30 min)
        4. 📈 Synthèse (10 min)

        **Total:** ~75 min 🚀
        """)


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    # === SIDEBAR NAVIGATION ===
    with st.sidebar:
        st.title("🔍 Explorateur")
        st.caption("Recherche Textuelle")

        st.markdown("### 📚 Navigation")

        # Navigation avec boutons stylés
        if "current_section" not in st.session_state:
            st.session_state.current_section = "🏠 Accueil"

        # Sections disponibles (désactiver Embeddings/Synthèse si pas installés)
        sections = ["🏠 Accueil", "📦 Datasets", "📊 TF-IDF", "🎯 BM25"]
        if EMBEDDINGS_AVAILABLE:
            sections.extend(["🧠 Embeddings", "📊 Synthèse"])
        else:
            sections.extend(["🧠 Embeddings 🔒", "📊 Synthèse 🔒"])

        for section_name in sections:
            # Style différent pour la section active
            if st.session_state.current_section == section_name:
                # Bouton actif (style différent)
                st.markdown(
                    f"""
                <div style="background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%); padding: 12px 20px; border-radius: 8px; margin-bottom: 8px; color: white; font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {section_name}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Bouton cliquable
                if st.button(
                    section_name, key=f"nav_{section_name}", use_container_width=True
                ):
                    st.session_state.current_section = section_name
                    st.rerun()

        section = st.session_state.current_section

        st.divider()

        # Configuration globale (si pas sur accueil)
        if section != "🏠 Accueil":
            st.markdown("### ⚙️ Configuration")

            # Sélection dataset
            dataset_names = [
                "recettes",
                "films",
                "wikipedia",
                "livres",
            ]  # Noms techniques
            dataset_labels = {
                "recettes": "🍝 Recettes",
                "films": "🎬 Films",
                "wikipedia": "📚 Wikipedia",
                "livres": "📖 Livres",
            }

            selected_dataset = st.selectbox(
                "Dataset:",
                dataset_names,
                format_func=lambda x: dataset_labels.get(x, x),
                key="dataset_select",
            )

            # Taille dataset
            use_extended = st.checkbox(
                "📦 Dataset étendu",
                value=False,
                help="Plus de documents pour tester performances",
                key="extended_check",
            )

            # Afficher la VRAIE taille du dataset sélectionné!
            try:
                # Utiliser get_dataset_info pour avoir les infos
                from src.data_loader import get_dataset_info

                dataset_info = get_dataset_info(selected_dataset)

                if use_extended:
                    estimated_docs = (
                        f"{dataset_info['size_extended']:,}"
                        if isinstance(dataset_info["size_extended"], int)
                        else dataset_info["size_extended"]
                    )
                    size_label = "(étendu)"
                else:
                    estimated_docs = f"{dataset_info['size_normal']:,}"
                    size_label = ""

                st.info(f"📊 **{estimated_docs} documents** {size_label}")

            except Exception:
                # Fallback en cas d'erreur
                estimated_docs = "~1,000" if use_extended else "~50"
                st.info(f"📊 {estimated_docs} documents")

            # Paramètres avancés
            with st.expander("🔧 Avancés"):
                remove_stopwords = st.checkbox(
                    "Supprimer stopwords", value=True, key="stopwords_check"
                )
                show_intermediate = st.checkbox(
                    "Calculs intermédiaires", value=False, key="intermediate_check"
                )

                # Menu de sélection du modèle d'embeddings (si disponible)
                if EMBEDDINGS_AVAILABLE:
                    st.markdown("**🧠 Modèle Embeddings**")

                    # Définir les modèles disponibles avec infos
                    embedding_models = {
                        "MiniLM-L6 (Petit, Rapide)": {
                            "name": "paraphrase-multilingual-MiniLM-L6-v2",
                            "size": "~80 MB",
                            "speed": "⚡⚡⚡",
                            "quality": "⭐⭐",
                        },
                        "MiniLM-L12 (Standard, Recommandé)": {
                            "name": "paraphrase-multilingual-MiniLM-L12-v2",
                            "size": "~120 MB",
                            "speed": "⚡⚡",
                            "quality": "⭐⭐⭐",
                        },
                        "MPNet (Grand, Meilleur)": {
                            "name": "paraphrase-multilingual-mpnet-base-v2",
                            "size": "~420 MB",
                            "speed": "⚡",
                            "quality": "⭐⭐⭐⭐",
                        },
                    }

                    selected_model_label = st.selectbox(
                        "Choisir un modèle:",
                        list(embedding_models.keys()),
                        index=1,  # Par défaut: MiniLM-L12 (recommandé)
                        key="embedding_model_select",
                        help="Petit = rapide mais moins précis | Grand = lent mais meilleur",
                    )

                    embedding_model_name = embedding_models[selected_model_label][
                        "name"
                    ]
                    model_info = embedding_models[selected_model_label]

                    # Afficher les infos du modèle sélectionné
                    st.caption(
                        f"📦 Taille: {model_info['size']} | Vitesse: {model_info['speed']} | Qualité: {model_info['quality']}"
                    )
                    st.caption("💾 Le modèle est téléchargé UNE FOIS et mis en cache!")
                else:
                    embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # Défaut si pas disponible

        st.divider()

        # Warning si embeddings pas disponibles
        if not EMBEDDINGS_AVAILABLE:
            st.warning("⚠️ Embeddings non installés. Sections verrouillées 🔒", icon="⚠️")

        st.caption("💡 Explore les sections pour apprendre!")

    # === ROUTING ===

    if section == "🏠 Accueil":
        render_home()

    elif section == "📦 Datasets":
        # Section d'exploration des datasets
        render_datasets_section(selected_dataset, use_extended)

    elif section in ["📊 TF-IDF", "🎯 BM25"]:
        # Charger le dataset
        with st.spinner("🔄 Chargement du dataset..."):
            start_load = time.time()
            dataset = load_cached_dataset(selected_dataset, extended=use_extended)
            load_time = time.time() - start_load

            documents_texts = [doc["text"] for doc in dataset]
            documents_titles = [doc["title"] for doc in dataset]
            documents_categories = [doc["category"] for doc in dataset]

        # Créer les engines
        if section == "📊 TF-IDF" or section == "🎯 BM25":
            with st.spinner("🧮 Préparation des moteurs de recherche..."):
                start_fit = time.time()
                tfidf_engine = create_tfidf_engine(
                    documents_texts, remove_stopwords=remove_stopwords
                )
                fit_time = time.time() - start_fit

        # Render la section appropriée
        if section == "📊 TF-IDF":
            render_tfidf_section(
                dataset,
                documents_texts,
                documents_titles,
                documents_categories,
                tfidf_engine,
                remove_stopwords,
                show_intermediate,
                load_time,
                fit_time,
            )

        elif section == "🎯 BM25":
            render_bm25_section(
                dataset,
                documents_texts,
                documents_titles,
                documents_categories,
                tfidf_engine,
                remove_stopwords,
            )

    elif section == "🧠 Embeddings" or section == "🧠 Embeddings 🔒":
        if EMBEDDINGS_AVAILABLE:
            # Charger dataset et engines comme pour BM25
            with st.spinner("🔄 Chargement du dataset..."):
                dataset = load_cached_dataset(selected_dataset, extended=use_extended)
                documents_texts = [doc["text"] for doc in dataset]
                documents_titles = [doc["title"] for doc in dataset]
                documents_categories = [doc.get("category", "Autre") for doc in dataset]

            # Créer TF-IDF et BM25 engines (pour comparaison)
            with st.spinner("⚙️ Initialisation des moteurs de recherche..."):
                tfidf_engine = create_tfidf_engine(documents_texts, remove_stopwords)
                bm25_engine = create_bm25_engine(documents_texts, remove_stopwords)

            # Appeler la vraie section Embeddings avec le modèle sélectionné
            render_embeddings_section(
                dataset,
                documents_texts,
                documents_titles,
                documents_categories,
                tfidf_engine,
                bm25_engine,
                remove_stopwords,
                embedding_model_name=embedding_model_name,  # NOUVEAU: modèle sélectionné!
            )
        else:
            st.title("🧠 Embeddings Vectoriels 🔒")
            st.error("""
            ### ⚠️ Module Non Disponible

            Les embeddings nécessitent **sentence-transformers** et **PyTorch**.

            **Pour installer:**
            ```bash
            pip install sentence-transformers torch transformers
            ```

            **Note:** L'installation peut prendre 5-10 minutes (plusieurs GB à télécharger).

            **En attendant**, tu peux utiliser **TF-IDF** et **BM25** qui sont 100% fonctionnels! 🚀
            """)

    elif section == "📊 Synthèse" or section == "📊 Synthèse 🔒":
        if EMBEDDINGS_AVAILABLE:
            # Charger dataset et tous les engines
            with st.spinner("🔄 Chargement du dataset..."):
                dataset = load_cached_dataset(selected_dataset, extended=use_extended)
                documents_texts = [doc["text"] for doc in dataset]
                documents_titles = [doc["title"] for doc in dataset]
                documents_categories = [doc.get("category", "Autre") for doc in dataset]

            # Créer TOUS les engines pour la synthèse
            with st.spinner("⚙️ Initialisation de tous les moteurs..."):
                tfidf_engine = create_tfidf_engine(documents_texts, remove_stopwords)
                bm25_engine = create_bm25_engine(documents_texts, remove_stopwords)
                # L'embedding engine sera créé dans render_synthesis_section si nécessaire

            # Appeler la vraie section Synthèse
            render_synthesis_section(
                dataset,
                documents_texts,
                documents_titles,
                documents_categories,
                tfidf_engine,
                bm25_engine,
                None,  # embedding_engine sera créé à la demande
            )
        else:
            st.title("📊 Synthèse Comparative 🔒")
            st.error("""
            ### ⚠️ Module Non Disponible

            La synthèse nécessite que **tous les moteurs** soient disponibles (TF-IDF, BM25, Embeddings).

            **Pour débloquer**, installe d'abord les embeddings:
            ```bash
            pip install sentence-transformers torch transformers
            ```

            **En attendant**, compare **TF-IDF vs BM25** dans la section BM25 → Comparaison! ⚔️
            """)

    # === FOOTER ===
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem 0;">
        <p>Créé avec ❤️ pour l'apprentissage de la recherche textuelle</p>
        <p style="font-size: 0.9rem;">📚 TF-IDF • 🎯 BM25 • 🧠 Embeddings (à venir)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
