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

from tfidf_engine import TFIDFEngine, preprocess_text, cosine_similarity
from bm25_engine import BM25Engine
from data_loader import load_dataset, get_all_datasets_info

# Imports optionnels pour Embeddings (nécessite sentence-transformers)
try:
    from embedding_engine import EmbeddingSearch
    from hybrid_search import HybridSearch

    EMBEDDINGS_AVAILABLE = True

    # Import des sections Embeddings et Synthèse (si dépendances disponibles)
    from app_embeddings_sections import (
        render_embeddings_section,
        create_embedding_engine,
        create_hybrid_engine,
    )
    from app_synthesis_sections import render_synthesis_section
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Le warning sera affiché dans la sidebar
from visualizations import (
    # TF-IDF visualizations
    plot_tf_comparison,
    plot_idf_curve,
    plot_idf_wordcloud,
    plot_tfidf_heatmap,
    plot_top_words_per_doc,
    plot_similarity_heatmap,
    plot_search_results,
    plot_documents_3d,
    plot_documents_2d,
    plot_tf_vs_tfidf_comparison,
    plot_vocabulary_stats,
    # BM25 visualizations
    plot_saturation_effect,
    plot_length_normalization,
    plot_parameter_space_heatmap,
    plot_tfidf_bm25_comparison,
    plot_score_distributions,
    # Embeddings visualizations
    plot_embedding_space_3d,
    plot_tsne_2d,
    plot_similarity_heatmap_embeddings,
    plot_clustering_2d,
    plot_technique_comparison_radar,
    plot_hybrid_alpha_effect,
    plot_multi_technique_comparison,
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


@st.cache_data
def load_cached_dataset(
    dataset_name: str, sample_size: int = None, extended: bool = False
):
    """Charge un dataset avec cache"""
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

    # Vérifier si on utilise HuggingFace ou hardcodé
    from data_loader import HF_AVAILABLE

    if HF_AVAILABLE:
        st.success("✅ **Hugging Face `datasets` est disponible!**")
    else:
        st.warning(
            "⚠️ **Hugging Face `datasets` NON disponible. Utilisation de données hardcodées.**"
        )

    st.info(f"""
    **Dataset actuel:** `{dataset_name}`
    **Taille:** `{"Extended (10k docs)" if use_extended else "Standard (1k docs)"}`
    **Documents chargés:** `{len(dataset)}`
    """)

    st.markdown("---")

    # === LISTE DES DOCUMENTS ===
    st.header("📋 Liste des Documents")

    # Recherche/Filtrage
    col1, col2 = st.columns([3, 1])
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

    st.caption(f"📊 {len(filtered_docs)} documents affichés (sur {len(dataset)} total)")

    # Sélecteur de document
    if len(filtered_docs) == 0:
        st.warning("😕 Aucun document trouvé avec ces filtres!")
    else:
        # Créer une liste de choix
        doc_choices = [
            f"[{i + 1}] {doc['title'][:60]}{'...' if len(doc['title']) > 60 else ''}"
            for i, doc in enumerate(filtered_docs)
        ]

        selected_idx = st.selectbox(
            "📄 Sélectionne un document à inspecter:",
            range(len(doc_choices)),
            format_func=lambda i: doc_choices[i],
        )

        # Afficher le document sélectionné
        if selected_idx is not None:
            doc = filtered_docs[selected_idx]

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
        '<p class="subtitle">Une application interactive pour maîtriser les techniques de recherche!</p>',
        unsafe_allow_html=True,
    )

    # === INTRO VISUELLE ===
    st.markdown("""
    ## 🎯 Qu'est-ce que la Recherche Textuelle?

    Imagine que tu as **10,000 recettes de cuisine** et tu cherches **"dessert au chocolat"**.
    Comment l'ordinateur trouve-t-il les **meilleurs résultats** parmi tous ces documents?

    C'est exactement ce que tu vas apprendre dans cette app! 🚀
    """)

    # === EXEMPLE CONCRET ===
    st.info("""
    **💡 Exemple Concret:**

    Tu tapes: **"pâtes italiennes fromage"**

    L'algorithme doit:
    1. Comprendre quels **mots sont importants** (pas "le", "la", "de"...)
    2. Trouver les documents qui **contiennent ces mots**
    3. **Classer** les résultats du plus au moins pertinent
    4. Te montrer les **meilleurs en premier**! 🎯
    """)

    st.markdown("---")

    # === SECTIONS DISPONIBLES (CARDS) ===
    st.markdown("## 📚 Parcours d'Apprentissage")

    # Section 1: TF-IDF
    with st.container():
        st.markdown("### 📊 Étape 1: TF-IDF - Les Fondamentaux")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("""
            **Niveau:** 🟢 Débutant
            **Durée:** 15-20 min
            **Concepts:** 5
            """)

        with col2:
            st.markdown("""
            **Term Frequency - Inverse Document Frequency**

            La technique **classique** de recherche textuelle. Tu apprendras:
            - ✅ Pourquoi compter les mots ne suffit pas
            - 📐 Comment normaliser les fréquences (TF)
            - 🔍 Pourquoi les mots rares sont plus importants (IDF)
            - 🧮 Comment calculer la similarité entre documents
            - ⚠️ Les limites de cette approche
            """)

        st.success("💡 **Recommandé:** Commence par TF-IDF pour comprendre les bases!")

    st.markdown("")

    # Section 2: BM25
    with st.container():
        st.markdown("### 🎯 Étape 2: BM25 - L'Amélioration")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("""
            **Niveau:** 🟡 Intermédiaire
            **Durée:** 20-25 min
            **Concepts:** 6
            """)

        with col2:
            st.markdown("""
            **Best Matching 25 - État de l'art**

            Une version **améliorée** de TF-IDF utilisée par les moteurs de recherche pro:
            - 🚀 Résout les problèmes de TF-IDF
            - 📈 Saturation intelligente (évite la sur-pondération)
            - 🎛️ Paramètres ajustables (k1, b) pour tuning
            - ⚔️ Comparaison directe avec TF-IDF
            - ✅ Meilleurs résultats en pratique
            """)

        st.info("🎓 **Prérequis:** Avoir compris TF-IDF avant!")

    st.markdown("")

    # Section 3: Embeddings
    if EMBEDDINGS_AVAILABLE:
        with st.container():
            st.markdown("### 🧠 Étape 3: Embeddings - La Sémantique")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("""
                **Niveau:** 🔴 Avancé
                **Durée:** 30-40 min
                **Concepts:** 7
                """)

            with col2:
                st.markdown("""
                **Recherche Sémantique par Réseaux de Neurones**

                La technique **moderne** basée sur l'IA:
                - 🤖 Comprend le **sens** des mots, pas juste leur présence
                - 🔄 Trouve des **synonymes** automatiquement
                - 🎯 Recherche par **concept** plutôt que par mot exact
                - 🌐 Utilise des modèles pré-entraînés (Sentence-BERT)
                - 🚀 Combinaison avec BM25 (Hybrid Search)
                """)

            st.success("🔥 **Bonus:** Compare les 3 techniques côte à côte!")

    else:
        st.warning("""
        ### 🧠 Embeddings 🔒

        Section non disponible - dépendances manquantes.
        Installe `sentence-transformers` pour débloquer cette section!

        ```bash
        pip install sentence-transformers torch
        ```
        """)

    st.markdown("---")

    # === GUIDE D'UTILISATION ===
    st.markdown("## 🚀 Guide d'Utilisation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1️⃣ Navigation

        Utilise la **sidebar** (←) pour:
        - Choisir une section
        - Sélectionner un dataset
        - Ajuster les paramètres
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ Exploration

        Dans chaque section:
        - 📖 **Intro:** Le concept expliqué
        - 🔢 **Concepts:** Formules détaillées
        - 🔍 **Recherche:** Teste en live
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ Apprentissage

        Profite de:
        - 📊 Graphiques interactifs
        - 🎓 Exemples pas-à-pas
        - ⚔️ Comparaisons entre techniques
        """)

    st.markdown("---")

    # === DATASETS ===
    st.markdown("## 📦 Datasets Disponibles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🍝 Recettes

        ~1,000 recettes de cuisine

        **Catégories:**
        - Italienne, Française
        - Asiatique, Mexicaine
        - Desserts, Plats

        **Idéal pour:** Recherches simples
        """)

    with col2:
        st.markdown("""
        ### 🎬 Films

        ~1,000 synopsis de films

        **Catégories:**
        - Science-fiction, Action
        - Comédie, Drame
        - Fantasy, Horreur

        **Idéal pour:** Concepts abstraits
        """)

    with col3:
        st.markdown("""
        ### 📚 Wikipedia

        ~1,000 articles variés

        **Catégories:**
        - Technologie, Histoire
        - Science, Sport
        - Culture, Géographie

        **Idéal pour:** Recherches complexes
        """)

    st.markdown("---")

    # === CALL TO ACTION ===
    st.markdown("""
    ## 🎓 Prêt à Apprendre?

    **Parcours recommandé:**

    1. 📊 **TF-IDF** → Comprends les bases (15 min)
    2. 🎯 **BM25** → Découvre les améliorations (20 min)
    3. 🧠 **Embeddings** → Explore l'IA moderne (30 min)
    4. 📈 **Synthèse** → Compare tout (10 min)

    **Temps total:** ~75 minutes pour maîtriser la recherche textuelle! 🚀
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(
            "👉 **Commence maintenant en sélectionnant TF-IDF dans la sidebar!**"
        )

    st.markdown("---")

    # === FOOTER ===
    st.caption("""
    💡 **Conseil:** Tu peux revenir sur cette page à tout moment en cliquant sur 🏠 Accueil dans la sidebar.

    📖 **Objectif pédagogique:** Cette app est conçue pour des étudiants en développement web (Bac+2/3) qui veulent comprendre comment fonctionnent les moteurs de recherche.
    """)


# ============================================================================
# SECTION TF-IDF (contenu existant restructuré)
# ============================================================================


def render_tfidf_section(
    dataset,
    documents_texts,
    documents_titles,
    documents_categories,
    engine,
    remove_stopwords,
    show_intermediate,
    load_time,
    fit_time,
):
    """Section TF-IDF complète avec tous les onglets"""

    st.title("📊 TF-IDF: Term Frequency - Inverse Document Frequency")

    # Sub-navigation avec boutons stylés
    tabs_tfidf = [
        "📖 Introduction",
        "🔢 Concepts",
        "🔍 Recherche",
        "📊 Exploration",
        "🎓 Pas-à-Pas",
        "⚡ Performance",
    ]
    tab = render_tab_navigation(tabs_tfidf, "tfidf_current_tab")

    if tab == "📖 Introduction":
        render_tfidf_intro()
    elif tab == "🔢 Concepts":
        render_tfidf_concepts(engine, documents_titles)
    elif tab == "🔍 Recherche":
        render_tfidf_search(
            engine,
            documents_texts,
            documents_titles,
            documents_categories,
            show_intermediate,
        )
    elif tab == "📊 Exploration":
        render_tfidf_exploration(engine, documents_titles, documents_categories)
    elif tab == "🎓 Pas-à-Pas":
        render_tfidf_stepbystep(
            documents_texts, documents_titles, documents_categories, remove_stopwords
        )
    elif tab == "⚡ Performance":
        render_tfidf_performance(
            engine, documents_texts, load_time, fit_time, remove_stopwords
        )


def render_tfidf_intro():
    """Introduction TF-IDF enrichie avec exemples détaillés"""
    st.header("📖 Introduction: Le Problème de la Recherche Textuelle")

    # === SECTION 1: LE CONTEXTE ===
    st.markdown("""
    ## 🌍 Le Contexte: Trouver l'aiguille dans la botte de foin

    Imagine que tu cherches **"recette italienne pâtes"** parmi 10,000 documents de cuisine.
    Comment l'ordinateur peut-il trouver les documents les plus **pertinents**?

    La solution naïve (compter les mots) échoue lamentablement. Voyons pourquoi! 👇
    """)

    st.divider()

    # === SECTION 2: L'ÉCHEC DE LA RECHERCHE NAÏVE ===
    st.markdown("### ❌ Problème #1: La Longueur des Documents")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        **Approche naïve:** Compter simplement le nombre d'occurrences du mot.

        #### Scénario Concret:

        Tu cherches **"chocolat"** dans des recettes:

        - **Doc A** (Titre: "Mousse au chocolat") - 50 mots
          - Mot "chocolat" apparaît **2 fois**
          - Proportion: **2/50 = 4%** du document
          - *C'est clairement une recette DE chocolat!*

        - **Doc B** (Titre: "Buffet complet") - 500 mots
          - Mot "chocolat" apparaît **3 fois** (mention rapide du dessert)
          - Proportion: **3/500 = 0.6%** du document
          - *Le chocolat est mentionné en passant*

        #### 💥 Le Bug:

        L'approche naïve dit: **Doc B est plus pertinent** (3 > 2 occurrences)

        La réalité: **Doc A est clairement meilleur!** (4% vs 0.6%)
        """)

    with col2:
        st.code(
            """
🔍 Recherche: "chocolat"

❌ Approche Naïve:
━━━━━━━━━━━━━━━━━━━━
Doc A: 2 occurrences
Doc B: 3 occurrences
Résultat: B > A ❌

✅ Approche Intelligente:
━━━━━━━━━━━━━━━━━━━━
Doc A: 4.0% du doc
Doc B: 0.6% du doc
Résultat: A > B ✅

💡 TF normalise par
   la longueur!
        """,
            language="text",
        )

    st.divider()

    # === SECTION 3: MOTS COMMUNS ===
    st.markdown("### ❌ Problème #2: Les Mots Communs Polluent Tout")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        #### Scénario: Recherche "cuisine traditionnelle"

        **Sans filtrage des mots communs:**

        Top 3 résultats naïfs:
        1. 📄 Doc avec **"la", "de", "un"** 50× chacun → Score énorme!
        2. 📄 Doc avec **"et", "dans", "avec"** 40× → Deuxième!
        3. 📄 Doc **vraiment** sur la cuisine traditionnelle → Troisième seulement!

        #### 💡 Le Problème:

        Les mots **super communs** comme "le", "la", "de", "un" apparaissent PARTOUT.

        Ils n'apportent **AUCUNE information** sur le sujet du document!

        - "le" → Présent dans 99% des documents → **Inutile!**
        - "traditionnelle" → Présent dans 2% des documents → **Très informatif!**

        #### 💡 La Solution:

        **IDF (Inverse Document Frequency)** pénalise les mots qui apparaissent partout.

        Plus un mot est rare dans le corpus, plus son **IDF est élevé**!
        """)

    with col2:
        st.code(
            """
🔍 "cuisine traditionnelle"

❌ Sans IDF:
━━━━━━━━━━━━━━
1. "la" (score: 150)
2. "de" (score: 120)
3. "un" (score: 100)
...
42. "traditionnelle"
    (score: 3)

✅ Avec IDF:
━━━━━━━━━━━━━━
IDF("la") = 0.01
  → 150 × 0.01 = 1.5

IDF("traditionnelle")
  = 3.2
  → 3 × 3.2 = 9.6

Résultat: "traditionnelle"
devient dominant! ✅
        """,
            language="text",
        )

    st.divider()

    # === SECTION 4: CAS D'USAGE RÉEL ===
    st.markdown("### 🎯 Cas d'Usage Réels de TF-IDF")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        #### 🔍 Moteurs de Recherche

        Google, Bing utilisaient TF-IDF avant les embeddings!

        **Exemple:**
        - Requête: "python tutorial"
        - TF-IDF trouve les docs qui parlent VRAIMENT de Python
        - Pas juste ceux qui mentionnent "python" 1 fois
        """)

    with col2:
        st.success("""
        #### 📧 Filtrage de Spam

        Détecter les emails frauduleux

        **Exemple:**
        - Spam: "GAGNEZ", "GRATUIT", "URGENT"
        - IDF faible (dans tous les spams)
        - Mais TF élevé dans spams
        - → Signature claire!
        """)

    with col3:
        st.warning("""
        #### 📊 Analyse de Documents

        Extraire les mots-clés d'un texte

        **Exemple:**
        - Article scientifique
        - TF-IDF extrait: "algorithme", "réseau", "neuronal"
        - Ignore: "est", "dans", "pour"
        - → Mots-clés automatiques!
        """)

    st.divider()

    # === SECTION 5: LA SOLUTION TF-IDF ===
    st.markdown("""
    ## ✅ La Solution: TF-IDF

    TF-IDF combine **deux mesures complémentaires** pour résoudre ces problèmes:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 📈 TF (Term Frequency)

        **"Fréquence locale du mot dans le document"**

        **Formule:**
        ```
        TF = (nombre d'occurrences) / (total mots doc)
        ```

        **Ce qu'il fait:**
        - ✅ Normalise par la longueur du document
        - ✅ Compare des docs courts et longs équitablement
        - ✅ Mesure l'importance locale d'un mot

        **Exemple:**
        - Doc de 100 mots avec "pizza" 5×
        - TF("pizza") = 5/100 = **0.05** (5%)
        """)

    with col2:
        st.success("""
        ### 📉 IDF (Inverse Document Frequency)

        **"Rareté globale du mot dans tout le corpus"**

        **Formule:**
        ```
        IDF = log(total_docs / docs_avec_mot)
        ```

        **Ce qu'il fait:**
        - ✅ Pénalise les mots très communs
        - ✅ Boost les mots rares et informatifs
        - ✅ Mesure l'importance globale

        **Exemple:**
        - "le": dans 9,900/10,000 docs
        - IDF("le") = log(10000/9900) ≈ **0.01**
        - "margherita": dans 50/10,000 docs
        - IDF("margherita") = log(10000/50) ≈ **5.3**
        """)

    st.divider()

    st.markdown("""
    ## 🧮 TF-IDF = TF × IDF

    La formule magique multiplie les deux mesures:

    ```
    TF-IDF(mot, doc) = TF(mot, doc) × IDF(mot, corpus)
    ```

    #### 💡 Interprétation:

    Un mot a un **TF-IDF élevé** si:
    1. Il apparaît **souvent dans CE document** (TF élevé) ET
    2. Il apparaît **rarement dans les autres documents** (IDF élevé)

    → C'est un mot **discriminant** pour ce document! 🎯
    """)

    # Exemple visuel
    with st.expander("📊 Voir un Exemple Complet Calculé"):
        st.markdown("""
        ### Exemple: 3 Documents sur la Cuisine

        **Corpus:**
        1. Doc A: "La pizza margherita est une pizza italienne"
        2. Doc B: "La pasta carbonara est une recette italienne"
        3. Doc C: "La cuisine italienne est délicieuse"

        **Calculs pour le mot "pizza" dans Doc A:**

        **1️⃣ TF (Term Frequency):**
        ```
        Doc A contient 8 mots, "pizza" apparaît 2 fois
        TF("pizza", Doc A) = 2 / 8 = 0.25
        ```

        **2️⃣ IDF (Inverse Document Frequency):**
        ```
        "pizza" apparaît dans 1 document sur 3
        IDF("pizza") = log(3 / 1) = log(3) ≈ 1.10
        ```

        **3️⃣ TF-IDF Final:**
        ```
        TF-IDF("pizza", Doc A) = 0.25 × 1.10 ≈ 0.275
        ```

        **Comparaison avec "la":**
        ```
        TF("la", Doc A) = 1 / 8 = 0.125
        IDF("la") = log(3 / 3) = 0  (présent partout!)
        TF-IDF("la", Doc A) = 0.125 × 0 = 0
        ```

        → **"pizza" a un score élevé, "la" est éliminé!** ✅
        """)

    st.divider()

    st.success("""
    ### 🎓 Dans les Prochaines Sections

    Tu vas découvrir:
    1. **Concepts TF-IDF** - Calculs détaillés avec visualisations
    2. **Recherche Interactive** - Teste le moteur en live!
    3. **Exploration du Corpus** - Analyse les mots-clés
    4. **Exemple Pas-à-Pas** - Déroule un calcul complet
    5. **Performance** - Complexité et optimisations

    **→ Passe à l'onglet suivant!** 👉
    """)


def render_tfidf_concepts(engine, documents_titles):
    """Concepts TF-IDF détaillés avec PÉDAGOGIE MAXIMALE"""
    st.header("🔢 Concepts TF-IDF en Profondeur")

    st.markdown("""
    TF-IDF se compose de **3 concepts fondamentaux** que nous allons explorer un par un.

    Chaque concept résout un problème spécifique de la recherche textuelle! 🎯
    """)

    # ============================================================================
    # CONCEPT 1: TERM FREQUENCY (TF)
    # ============================================================================
    with st.expander(
        "📈 **1. Term Frequency (TF)** - Fréquence des Mots", expanded=True
    ):
        st.markdown("""
        ### 💡 L'Intuition

        **"Si un mot apparaît souvent dans un document, ce document parle probablement de ce sujet"**

        ### 🤔 Le Problème à Résoudre

        Imagine deux documents qui parlent de "chocolat":
        - **Doc A** (50 mots): "chocolat" apparaît **2 fois**
        - **Doc B** (500 mots): "chocolat" apparaît **3 fois**

        Sans normalisation, Doc B semble plus pertinent (3 > 2).
        **Mais!** Doc A consacre **4%** de son contenu au chocolat (2/50), tandis que Doc B seulement **0.6%** (3/500)!

        ### 📐 La Formule
        """)

        st.latex(
            r"\text{TF}(mot, doc) = \frac{\text{nombre d'occurrences}}{\text{total de mots dans le doc}}"
        )

        st.markdown("""
        **Pourquoi diviser?** Pour normaliser! Un document court avec 2 occurrences peut être plus "à propos"
        du sujet qu'un document long avec 5 occurrences.

        ### 📊 Exemple Visuel sur Notre Corpus

        Voici les TF de quelques mots dans 3 documents:
        """)

        # Graphique RÉDUIT (colonnes pour prendre moins d'espace!)
        col1, col2 = st.columns([2, 1])

        with col1:
            sample_indices = [0, 1, 2]
            sample_titles = [documents_titles[i] for i in sample_indices]
            fig_tf = plot_tf_comparison(engine.documents, sample_indices, sample_titles)
            st.pyplot(fig_tf)

        with col2:
            st.markdown("""
            **🔍 Comment lire ce graphique:**

            - **Hauteur des barres** = TF (fréquence normalisée)
            - **Plus haut** = mot plus fréquent dans ce doc
            - **Comparaison** entre docs pour le même mot

            **💡 Observation:**

            Un mot peut avoir un TF élevé dans un doc et faible dans un autre.

            **Exemple:** "pâtes" a un TF de 0.08 dans la recette italienne, mais 0.00 dans le film!

            ➡️ Le TF capture bien le **sujet local** du document! ✅
            """)

        st.info("""
        **✅ Ce que TF résout:** Compare les documents équitablement, peu importe leur longueur!

        **⚠️ Ce que TF ne résout PAS:** Les mots communs ("le", "la", "de") ont aussi des TF élevés...
        On verra comment IDF règle ce problème! 👇
        """)

    # ============================================================================
    # CONCEPT 2: INVERSE DOCUMENT FREQUENCY (IDF)
    # ============================================================================
    with st.expander("📉 **2. Inverse Document Frequency (IDF)** - Rareté des Mots"):
        st.markdown("""
        ### 💡 L'Intuition

        **"Un mot RARE est plus INFORMATIF qu'un mot commun"**

        ### 🤔 Le Problème à Résoudre

        Tous les mots ne sont PAS égaux!
        - Le mot **"le"** apparaît dans TOUS les documents → **PEU informatif** 😐
        - Le mot **"carbonara"** apparaît dans 1 seul document → **TRÈS informatif**! 🎯

        ### 📐 La Formule
        """)

        st.latex(r"\text{IDF}(mot) = \log\left(\frac{N}{n}\right) + 1")

        st.caption(
            "Où: N = nombre total de documents, n = nombre de documents contenant le mot"
        )

        st.markdown("""
        **Pourquoi le logarithme?** Pour compresser l'échelle! Sans log, un mot présent dans 1 doc sur 10,000
        aurait un IDF de 10,000 - bien trop élevé!

        Le log transforme ça en ~4, plus raisonnable. 📉

        ### 📊 Exemple: Courbe IDF

        Voici comment l'IDF varie selon le nombre de documents contenant un mot:
        """)

        # Graphiques IDF en colonnes
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Courbe IDF vs Fréquence**")
            fig_idf = plot_idf_curve(
                engine.idf_vector, engine.vocabulary, engine.documents
            )
            st.pyplot(fig_idf)

            st.markdown("""
            **🔍 Comment lire:**
            - **Axe X** = Nombre de docs contenant le mot
            - **Axe Y** = Score IDF
            - **Courbe décroissante** = Plus un mot est fréquent, plus son IDF est faible

            **💡 Observation:**
            - Mot dans **1 doc** → IDF élevé (~3)
            - Mot dans **TOUS les docs** → IDF proche de 0
            """)

        with col2:
            st.markdown("**☁️ WordCloud par IDF**")
            # Prendre les 200 premiers mots du vocabulaire
            idf_dict = {
                engine.vocabulary[i]: engine.idf_vector[i]
                for i in range(min(200, len(engine.vocabulary)))
            }
            fig_wc = plot_idf_wordcloud(idf_dict)
            st.pyplot(fig_wc)

            st.markdown("""
            **🔍 Comment lire:**
            - **Taille du mot** = IDF (rareté)
            - **Gros mots** = mots RARES (informatifs!)
            - **Petits mots** = mots communs (peu informatifs)

            **💡 Observation:**

            Les mots spécifiques sont **gros** (ex: "carbonara", "tiramisu"), tandis que les mots génériques sont **petits** (ex: "très", "bien").
            """)

        st.success("""
        **✅ Ce que IDF résout:** Donne plus de poids aux mots RARES (informatifs) et moins aux mots COMMUNS!

        **Exemple concret:**
        - "le" → IDF = 0.05 (commun, peu informatif)
        - "carbonara" → IDF = 2.5 (rare, très informatif!)

        ➡️ Maintenant combinons TF et IDF! 🎯
        """)

    # ============================================================================
    # CONCEPT 3: TF-IDF COMBINÉ
    # ============================================================================
    with st.expander("🎯 **3. TF-IDF Combiné** - La Magie Opère!"):
        st.markdown("""
        ### 💡 L'Idée Géniale

        **TF-IDF = Multiplie la fréquence locale (TF) par la rareté globale (IDF)**

        ### 📐 La Formule Finale
        """)

        st.latex(
            r"\text{TF-IDF}(mot, doc) = \text{TF}(mot, doc) \times \text{IDF}(mot)"
        )

        st.markdown("""
        **Ce que ça donne:**

        Les mots avec un **TF-IDF élevé** sont ceux qui sont:
        1. **Fréquents dans LE document** (TF élevé) ✅
        2. **Rares dans LES AUTRES documents** (IDF élevé) ✅

        ➡️ Ce sont exactement les mots qui **caractérisent** ce document! 🎯

        ### 📊 Heatmap TF-IDF

        Visualisation des mots les plus importants pour chaque document:
        """)

        st.info("""
        **💡 Avant de regarder la heatmap:**

        - **Lignes** = Documents
        - **Colonnes** = Mots (top 15)
        - **Couleur** = Score TF-IDF (rouge = élevé, bleu = faible)

        **Ce qu'on cherche:** Des cases **rouges** qui montrent quel mot caractérise quel document!
        """)

        # Heatmap réduite
        col1, col2 = st.columns([3, 1])

        with col1:
            fig_heatmap = plot_tfidf_heatmap(
                engine.tfidf_matrix, engine.vocabulary, documents_titles, top_words=15
            )
            st.pyplot(fig_heatmap)

        with col2:
            st.markdown("""
            **🔍 Comment analyser:**

            1. **Regarder les colonnes** (mots):
               - Certains mots sont rouges pour UN doc, bleus pour les autres
               - ➡️ Ce mot CARACTÉRISE ce doc!

            2. **Regarder les lignes** (docs):
               - Chaque doc a ses propres mots "rouges"
               - ➡️ Son "empreinte" unique!

            3. **Patterns intéressants**:
               - Docs similaires ont des patterns similaires
               - Docs différents ont patterns différents

            **Exemple:**
            - Doc "Pâtes Carbonara" → "pâtes", "parmesan", "guanciale" en rouge
            - Doc "Interstellar" → "espace", "temps", "trou" en rouge

            ➡️ TF-IDF capture parfaitement le sujet de chaque doc! ✅
            """)

        st.success("""
        **🎉 Félicitations! Tu comprends TF-IDF!**

        **Récap en 3 points:**
        1. **TF** = Fréquence normalisée (local au document)
        2. **IDF** = Rareté (global au corpus)
        3. **TF-IDF** = TF × IDF = Mots qui caractérisent chaque document!

        **🎯 Utilisation:** Pour comparer une **requête** avec des **documents**,
        on calcule le TF-IDF de chaque mot, puis on mesure la **similarité** (prochain concept!)
        """)

    # ============================================================================
    # CONCEPT BONUS: COSINE SIMILARITY
    # ============================================================================
    with st.expander("📐 **Bonus: Similarité Cosinus** - Comparer les Documents"):
        st.markdown("""
        ### 💡 Le Concept Géométrique

        Une fois qu'on a les vecteurs TF-IDF, comment comparer une **requête** avec des **documents**?

        **Réponse:** La Similarité Cosinus! Elle mesure l'**angle** entre deux vecteurs.

        ### 📐 La Formule
        """)

        st.latex(r"\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}")

        st.markdown("""
        **Composantes:**
        - **A · B** = Produit scalaire (dot product) des vecteurs
        - **||A||** = Norme (longueur) du vecteur A
        - **||B||** = Norme (longueur) du vecteur B

        **Résultat:** Un score entre **0** et **1**:
        - **1.0** = Vecteurs identiques (angle = 0°) → **Très similaires!** 🎯
        - **0.5** = Vecteurs à 60° → Moyennement similaires
        - **0.0** = Vecteurs perpendiculaires (90°) → Pas similaires du tout

        ### 🤔 Pourquoi l'Angle et pas juste la Distance?

        **Exemple concret:**
        - Doc A (court): Vecteur TF-IDF [0.1, 0.2, 0.1]
        - Doc B (long): Vecteur TF-IDF [0.5, 1.0, 0.5]

        Ces vecteurs pointent dans la **même direction** (ratio 1:2:1), mais B est 5× plus long!

        - **Distance euclidienne:** Grande! ❌ (suggère qu'ils sont différents)
        - **Angle (cosinus):** Petit! ✅ (détecte qu'ils parlent du même sujet)

        ➡️ L'angle capture la **similitude thématique** indépendamment de la longueur! 🎯
        """)

        st.info("""
        **💡 En pratique:**

        Pour une requête "plat italien pâtes":
        1. Calculer son vecteur TF-IDF
        2. Calculer la similarité cosinus avec CHAQUE document
        3. Trier les documents par score décroissant
        4. Afficher les top résultats!

        **C'est exactement ce que fait la section "Recherche Interactive"!** 🔍
        """)

    st.markdown("---")
    st.success("""
    ✅ **Section Concepts terminée!**

    Tu maîtrises maintenant:
    - TF (fréquence normalisée)
    - IDF (rareté globale)
    - TF-IDF (combinaison magique)
    - Similarité Cosinus (comparaison)

    **👉 Passe à la "Recherche Interactive" pour voir TF-IDF en action!**
    """)


def render_tfidf_search(
    engine, documents_texts, documents_titles, documents_categories, show_intermediate
):
    """Recherche interactive TF-IDF avec analyses pédagogiques"""
    st.header("🔍 Recherche Interactive TF-IDF")

    st.markdown("""
    **Teste TF-IDF en action!** 🚀

    Entre une requête (plusieurs mots), et on va trouver les documents les plus pertinents
    en calculant la **similarité cosinus** entre ta requête et chaque document.

    **Comment ça marche:**
    1. Ta requête est transformée en vecteur TF-IDF
    2. On calcule la similarité avec TOUS les documents
    3. On trie par score décroissant
    4. On affiche les meilleurs résultats! 🎯
    """)

    # Utiliser un formulaire pour soumission avec Enter
    with st.form("tfidf_search_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "🔎 Entre ta requête:",
                value="plat italien pâtes",  # Valeur par défaut!
                placeholder="Ex: plat italien, science-fiction espace, guerre mondiale...",
                key="tfidf_query_input",
                help='💡 **Exemples:** "plat italien pâtes fromage" | "cuisine asiatique épicée crevettes" | "dessert chocolat français" | "poisson grillé méditerranéen"',
            )

        with col2:
            top_k = st.slider(
                "Résultats:",
                3,
                20,
                5,
                key="tfidf_topk_slider",
                help="Nombre de documents les plus pertinents à afficher",
            )

        # Bouton de soumission (Enter fonctionne aussi!)
        submitted = st.form_submit_button("🚀 Rechercher!", type="primary")

    if submitted and query:
        with st.spinner("🔍 Recherche en cours..."):
            results = engine.search(query, top_k=top_k)

            if len(results) == 0 or all(score == 0 for _, score in results):
                st.warning("😕 Aucun résultat. Essaie d'autres mots!")
            else:
                st.success(f"✅ {len(results)} résultats trouvés!")

                # ========= GRAPHIQUE + ANALYSE CÔTE À CÔTE =========
                st.markdown("### 📊 Visualisation des Scores")

                col_graph, col_analysis = st.columns([2, 1])

                with col_graph:
                    fig_results = plot_search_results(results, documents_titles, query)
                    st.pyplot(fig_results)

                with col_analysis:
                    st.markdown("**🔍 Comment lire ce graphique:**")
                    st.markdown("""
                    - **Axe X** = Score de similarité (0 à 1)
                    - **Axe Y** = Documents trouvés
                    - **Plus à droite** = plus similaire!

                    **💡 Interprétation des scores:**
                    - **> 0.5** → Très pertinent! 🎯
                    - **0.3 - 0.5** → Moyennement pertinent 👌
                    - **< 0.3** → Faiblement pertinent 😐
                    """)

                    # Analyse automatique des résultats!
                    top_score = results[0][1]
                    score_range = results[0][1] - results[-1][1]

                    if top_score > 0.5:
                        st.success(
                            f"🎯 **Excellent!** Le top résultat a un score de {top_score:.3f} - très pertinent!"
                        )
                    elif top_score > 0.3:
                        st.info(
                            f"👌 **Bon!** Score de {top_score:.3f} - pertinence moyenne."
                        )
                    else:
                        st.warning(
                            f"😐 **Moyen...** Score max de {top_score:.3f} - essaye d'autres mots?"
                        )

                    if score_range > 0.2:
                        st.markdown(
                            f"📊 **Bonne séparation:** Les scores varient de {results[-1][1]:.3f} à {results[0][1]:.3f} - TF-IDF distingue bien les docs!"
                        )
                    else:
                        st.markdown(
                            f"📊 **Scores proches:** Écart de seulement {score_range:.3f} - les docs se ressemblent!"
                        )

                # ========= RÉSULTATS DÉTAILLÉS =========
                st.markdown("---")
                st.markdown("### 🎯 Résultats Détaillés")

                for rank, (doc_idx, score) in enumerate(results[:5], 1):
                    # Badge de qualité selon le score
                    if score > 0.5:
                        badge = "🔥 **Très pertinent!**"
                        badge_color = "green"
                    elif score > 0.3:
                        badge = "👌 **Pertinent**"
                        badge_color = "blue"
                    else:
                        badge = "😐 **Faiblement pertinent**"
                        badge_color = "orange"

                    with st.expander(
                        f"**#{rank}** - {documents_titles[doc_idx]} • Score: **{score:.3f}** {badge}"
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.caption(
                                f"**Catégorie:** {documents_categories[doc_idx]}"
                            )
                            st.write(documents_texts[doc_idx][:300] + "...")

                        with col2:
                            st.markdown("**📊 Pourquoi ce score?**")

                            # Analyser les mots de la query présents dans le doc
                            query_words = set(query.lower().split())
                            doc_words = set(documents_texts[doc_idx].lower().split())
                            common_words = query_words & doc_words

                            if common_words:
                                st.markdown(
                                    f"✅ **Mots en commun:** {', '.join(list(common_words)[:5])}"
                                )
                                st.markdown(
                                    f"📈 **Overlap:** {len(common_words)}/{len(query_words)} mots"
                                )
                            else:
                                st.markdown("❌ Aucun mot en commun (synonymes?)")

                            # Optionnel: afficher les calculs détaillés
                            if show_intermediate:
                                with st.expander("🔬 Calculs détaillés"):
                                    explanation = engine.get_explanation(query, doc_idx)
                                    st.json(explanation)

                # ========= CONSEILS PÉDAGOGIQUES =========
                st.markdown("---")
                st.info("""
                **💡 Expérimente!**

                - **Requête courte** (1-2 mots) → Résultats larges
                - **Requête longue** (4-5 mots) → Résultats précis
                - **Mots rares** → Meilleurs scores (IDF élevé!)
                - **Mots communs** → Scores plus faibles

                **🎯 Astuce:** Utilise des mots **spécifiques** à ce que tu cherches!
                """)


def render_tfidf_exploration(engine, documents_titles, documents_categories):
    """Exploration du corpus TF-IDF avec analyses approfondies"""
    st.header("📊 Exploration du Corpus")

    st.markdown("""
    Cette section te permet d'explorer le **corpus dans son ensemble** et de comprendre
    ses caractéristiques globales! 🔬

    Tu verras:
    - Les statistiques du corpus
    - La distribution du vocabulaire
    - Les mots les plus informatifs (IDF élevé)
    - La structure des documents en 3D
    """)

    # ============================================================================
    # MÉTRIQUES GLOBALES
    # ============================================================================
    st.markdown("### 📈 Métriques du Corpus")

    col1, col2, col3, col4 = st.columns(4)

    num_docs = len(documents_titles)
    vocab_size = len(engine.vocabulary)
    avg_words = np.mean([len(doc) for doc in engine.documents])
    num_categories = len(set(documents_categories))

    col1.metric(
        "📚 Documents", num_docs, help="Nombre total de documents dans le corpus"
    )
    col2.metric(
        "🔤 Vocabulaire",
        vocab_size,
        help="Nombre de mots uniques (après preprocessing)",
    )
    col3.metric(
        "📝 Mots/Doc", f"{avg_words:.1f}", help="Longueur moyenne d'un document"
    )
    col4.metric("🏷️ Catégories", num_categories, help="Nombre de catégories différentes")

    # Interprétation automatique
    st.markdown("**💡 Interprétation:**")

    if vocab_size > num_docs * 10:
        st.info(
            f"📖 **Vocabulaire riche:** {vocab_size} mots pour {num_docs} docs → Corpus diversifié!"
        )
    elif vocab_size > num_docs * 5:
        st.info(f"📖 **Vocabulaire normal:** Ratio vocabulaire/docs équilibré.")
    else:
        st.warning(
            f"📖 **Vocabulaire limité:** Peu de mots uniques → Docs probablement similaires."
        )

    if avg_words > 100:
        st.info(
            f"📄 **Documents longs:** Moyenne de {avg_words:.0f} mots → Textes détaillés!"
        )
    elif avg_words > 50:
        st.info(f"📄 **Documents moyens:** Longueur raisonnable pour l'analyse.")
    else:
        st.info(
            f"📄 **Documents courts:** {avg_words:.0f} mots en moyenne → Textes concis!"
        )

    st.markdown("---")

    # ============================================================================
    # DISTRIBUTION DU VOCABULAIRE
    # ============================================================================
    st.markdown("### 📊 Distribution du Vocabulaire")

    col_graph1, col_analysis1 = st.columns([2, 1])

    with col_graph1:
        st.markdown("**📈 Statistiques de Fréquence**")
        fig_vocab = plot_vocabulary_stats(engine.documents)
        st.pyplot(fig_vocab)

    with col_analysis1:
        st.markdown("**🔍 Comment lire:**")
        st.markdown("""
        Ce graphique montre la **distribution des longueurs de documents**.

        - **Axe X** = Longueur du document (nombre de mots)
        - **Axe Y** = Nombre de documents
        - **Forme de la courbe** = Distribution du corpus

        **💡 Ce qu'on veut:**
        - **Distribution uniforme** → Corpus équilibré ✅
        - **Pics multiples** → Catégories distinctes 🎯
        - **Un seul pic** → Docs similaires en longueur

        **📊 Observation:**

        Si tous les docs ont ~la même longueur, TF-IDF fonctionnera bien!
        Si les longueurs varient beaucoup, attention à la normalisation!
        """)

    st.markdown("---")

    # ============================================================================
    # TOP MOTS PAR IDF
    # ============================================================================
    st.markdown("### 🏆 Top Mots les Plus Informatifs (IDF)")

    st.markdown("""
    Voici les mots avec les **IDF les plus élevés** - ce sont les mots les plus **RARES** et donc
    les plus **INFORMATIFS** du corpus! 🎯
    """)

    # Extraire top 20 mots par IDF
    idf_items = [
        (engine.vocabulary[idx], engine.idf_vector[idx])
        for idx in range(len(engine.vocabulary))
    ]
    top_idf = sorted(idf_items, key=lambda x: x[1], reverse=True)[:20]

    col_graph2, col_analysis2 = st.columns([2, 1])

    with col_graph2:
        st.markdown("**📊 Top 20 Mots par IDF**")

        # Créer un bar chart simple
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        words = [w for w, _ in top_idf]
        idfs = [idf for _, idf in top_idf]
        ax.barh(words[::-1], idfs[::-1], color="#1f77b4")
        ax.set_xlabel("Score IDF")
        ax.set_title("Mots les Plus Informatifs")
        plt.tight_layout()
        st.pyplot(fig)

    with col_analysis2:
        st.markdown("**🔍 Analyse:**")
        st.markdown(f"""
        **Top 3 mots:**
        1. **{top_idf[0][0]}** ({top_idf[0][1]:.2f})
        2. **{top_idf[1][0]}** ({top_idf[1][1]:.2f})
        3. **{top_idf[2][0]}** ({top_idf[2][1]:.2f})

        **💡 Ce que ça signifie:**

        Ces mots sont **rares** dans le corpus!
        - IDF élevé → Peu de docs contiennent ce mot
        - ➡️ Très informatif pour caractériser un doc

        **🎯 En pratique:**

        Si ta requête contient ces mots, les résultats seront **très précis**!

        Si un document contient ces mots, il se **démarque** des autres!
        """)

    st.markdown("---")

    # ============================================================================
    # PROJECTION 3D DES DOCUMENTS
    # ============================================================================
    st.markdown("### 🌐 Projection 3D des Documents")

    st.info("""
    **💡 Avant de regarder la visualisation:**

    Chaque document est représenté par un **point dans l'espace 3D**.
    - La position est calculée avec **PCA** (réduction de dimensionalité)
    - Les couleurs = catégories
    - **Documents proches** = sujets similaires!
    - **Documents éloignés** = sujets différents!

    **🎯 Ce qu'on cherche:**
    - Des **clusters** (groupes) par catégorie ✅
    - Une bonne **séparation** entre catégories ✅
    """)

    col_graph3, col_analysis3 = st.columns([3, 1])

    with col_graph3:
        fig_3d = plot_documents_3d(
            engine.tfidf_matrix, documents_titles, documents_categories
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_analysis3:
        st.markdown("**🔍 Interprétation:**")
        st.markdown("""
        **Comment analyser:**

        1. **Rotation:** Clique et fais glisser pour tourner la vue 🔄

        2. **Zoom:** Scroll pour zoomer/dézoomer 🔍

        3. **Hover:** Survole un point pour voir le titre 👆

        **💡 Patterns à observer:**

        - **Clusters bien séparés** → TF-IDF distingue bien les catégories! ✅

        - **Chevauchement** → Certains docs se ressemblent malgré des catégories différentes 🤔

        - **Points isolés** → Docs uniques, différents des autres! 🌟

        **🎯 Utilité:**

        Cette visualisation montre si ton corpus est bien **structuré** et si TF-IDF capte les **différences** entre documents!
        """)

    st.markdown("---")
    st.success("""
    ✅ **Section Exploration terminée!**

    Tu as maintenant une **vue d'ensemble complète** du corpus:
    - Ses statistiques globales
    - Ses mots les plus informatifs
    - Sa structure spatiale

    **👉 Ces analyses t'aident à comprendre si TF-IDF est adapté à ton corpus!**
    """)


def render_tfidf_stepbystep(
    documents_texts, documents_titles, documents_categories, remove_stopwords
):
    """Exemple pas-à-pas TF-IDF COMPLET avec tous les calculs détaillés"""
    st.header("🎓 Exemple Complet Pas-à-Pas")

    st.markdown("""
    Dans cette section, tu vas voir **TOUS les calculs** en détail, étape par étape!

    On va prendre 3 documents et calculer leur similarité avec ta requête. 🔬
    """)

    # === DOCUMENTS D'EXEMPLE ===
    sample_indices = list(range(min(3, len(documents_texts))))

    st.markdown("### 📚 Documents utilisés pour l'exemple")

    for idx in sample_indices:
        with st.expander(
            f"📄 Document {idx + 1}: {documents_titles[idx]}", expanded=(idx == 0)
        ):
            st.write(f"**Catégorie:** {documents_categories[idx]}")
            st.write(f"**Contenu:** {documents_texts[idx]}")
            word_count = len(documents_texts[idx].split())
            st.caption(f"📊 Longueur: {word_count} mots")

    st.markdown("---")

    # === QUERY INPUT ===
    query = st.text_input(
        "🔎 Ta requête de test:",
        value="plat italien fromage",
        key="tfidf_tutorial",
        help='💡 **Exemples:** "chocolat dessert" | "pâtes italiennes sauce" | "poisson grillé citron"',
    )

    if not query:
        st.warning("⬆️ Entre une requête ci-dessus pour voir les calculs!")
        return

    # === CALCULS ===
    with st.spinner("🧮 Calcul en cours..."):
        sample_texts = [documents_texts[i] for i in sample_indices]
        mini_engine = TFIDFEngine(sample_texts, remove_stopwords=remove_stopwords)
        mini_engine.fit()

        query_tokens = preprocess_text(query)

        st.success(f'✅ Calculs terminés pour la requête: **"{query}"**')

    # === ÉTAPE 1: VOCABULAIRE ===
    st.markdown("---")
    st.markdown("## 🔢 Étape 1: Construction du Vocabulaire")

    st.markdown("""
    On commence par **extraire tous les mots uniques** de nos 3 documents.
    C'est notre **vocabulaire** (ou *vocabulary*).
    """)

    vocab_size = len(mini_engine.vocabulary)
    st.metric("📚 Taille du vocabulaire", f"{vocab_size} mots uniques")

    with st.expander("👀 Voir le vocabulaire complet"):
        vocab_list = sorted(list(mini_engine.vocabulary))
        st.write(", ".join(vocab_list[:100]))
        if len(vocab_list) > 100:
            st.caption(f"... et {len(vocab_list) - 100} autres mots")

    # === ÉTAPE 2: TERM FREQUENCY (TF) ===
    st.markdown("---")
    st.markdown("## 📊 Étape 2: Calcul des Term Frequencies (TF)")

    st.markdown("""
    **TF = Combien de fois un mot apparaît dans un document, normalisé par la longueur.**

    **Formule:** `TF(mot, doc) = nb_occurrences / nb_total_mots`

    **Pourquoi normaliser?** Pour ne pas favoriser les documents longs!
    """)

    st.latex(r"\text{TF}(t, d) = \frac{\text{count}(t, d)}{|\text{words}(d)|}")

    # Calculer TF pour les mots de la query
    query_words_in_vocab = [w for w in query_tokens if w in mini_engine.vocabulary]

    if len(query_words_in_vocab) == 0:
        st.warning(
            "⚠️ Aucun mot de ta requête n'est dans le vocabulaire! Essaie d'autres mots."
        )
        return

    st.info(
        f"🎯 **Mots de ta requête dans le vocabulaire:** {', '.join(query_words_in_vocab)}"
    )

    # Créer tableau TF
    tf_data = []
    for doc_idx in sample_indices:
        row = {"Document": documents_titles[doc_idx][:30] + "..."}
        for word in query_words_in_vocab:
            word_idx = mini_engine.word_to_idx[word]  # FIX: Utiliser word_to_idx!
            tf_value = mini_engine.tf_matrix[doc_idx, word_idx]
            row[word] = f"{tf_value:.4f}"
        tf_data.append(row)

    df_tf = pd.DataFrame(tf_data)
    st.markdown("**📊 Tableau des TF (Term Frequencies):**")
    st.dataframe(df_tf, use_container_width=True, hide_index=True)

    st.markdown("""
    **💡 Interprétation:**
    - Plus le TF est **élevé**, plus le mot est **fréquent** dans le document
    - Un TF de 0.05 = le mot représente **5%** du document
    - Un TF de 0.00 = le mot n'apparaît **pas** dans ce document
    """)

    # === ÉTAPE 3: INVERSE DOCUMENT FREQUENCY (IDF) ===
    st.markdown("---")
    st.markdown("## 🔍 Étape 3: Calcul des Inverse Document Frequencies (IDF)")

    st.markdown("""
    **IDF = Mesure de la rareté d'un mot dans TOUS les documents.**

    **Formule:** `IDF(mot) = log(nb_total_docs / nb_docs_contenant_mot)`

    **Pourquoi?** Les mots **rares** sont plus **informatifs** que les mots communs!
    """)

    st.latex(r"\text{IDF}(t) = \log\left(\frac{N}{n_t}\right)")

    st.caption(
        "Où: N = nombre total de documents, n_t = nombre de documents contenant le terme t"
    )

    # Calculer IDF pour les mots de la query
    idf_data = []
    for word in query_words_in_vocab:
        word_idx = mini_engine.word_to_idx[word]  # FIX: Utiliser word_to_idx!
        idf_value = mini_engine.idf_vector[word_idx]

        # Compter dans combien de docs le mot apparaît
        docs_with_word = sum(
            1
            for doc_idx in sample_indices
            if mini_engine.tf_matrix[doc_idx, word_idx] > 0
        )

        idf_data.append(
            {
                "Mot": word,
                "Docs contenant": f"{docs_with_word}/{len(sample_indices)}",
                "IDF": f"{idf_value:.4f}",
                "Rareté": "🔴 Rare"
                if docs_with_word == 1
                else "🟡 Moyen"
                if docs_with_word == 2
                else "🟢 Commun",
            }
        )

    df_idf = pd.DataFrame(idf_data)
    st.markdown("**📊 Tableau des IDF (Inverse Document Frequencies):**")
    st.dataframe(df_idf, use_container_width=True, hide_index=True)

    st.markdown("""
    **💡 Interprétation:**
    - IDF **élevé** (ex: 0.48) = mot **RARE** (apparaît dans peu de docs) → **très informatif**! 🔴
    - IDF **moyen** (ex: 0.18) = mot **commun** dans certains docs → informatif 🟡
    - IDF **faible** (ex: 0.00) = mot dans **TOUS** les docs → peu informatif 🟢
    """)

    # === ÉTAPE 4: TF-IDF (MULTIPLICATION) ===
    st.markdown("---")
    st.markdown("## 🎯 Étape 4: Calcul Final TF-IDF")

    st.markdown("""
    **TF-IDF = TF × IDF**

    On **multiplie** la fréquence locale (TF) par la rareté globale (IDF)!

    **Résultat:** Les mots qui sont à la fois:
    - **Fréquents dans le document** (TF élevé)
    - **Rares dans le corpus** (IDF élevé)

    ... ont un **score TF-IDF élevé**! C'est eux qui caractérisent le document! ✨
    """)

    st.latex(r"\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)")

    # Créer tableau TF-IDF
    tfidf_data = []
    for doc_idx in sample_indices:
        row = {"Document": documents_titles[doc_idx][:30] + "..."}
        for word in query_words_in_vocab:
            word_idx = mini_engine.word_to_idx[word]  # FIX: Utiliser word_to_idx!
            tfidf_value = mini_engine.tfidf_matrix[doc_idx, word_idx]
            row[word] = f"{tfidf_value:.4f}"
        tfidf_data.append(row)

    df_tfidf = pd.DataFrame(tfidf_data)
    st.markdown("**📊 Tableau des TF-IDF:**")
    st.dataframe(df_tfidf, use_container_width=True, hide_index=True)

    # === ÉTAPE 5: VECTORISATION DE LA QUERY ===
    st.markdown("---")
    st.markdown("## 🔤 Étape 5: Vectorisation de la Requête")

    st.markdown("""
    On doit aussi calculer le **vecteur TF-IDF de la requête**!

    **Processus:**
    1. Calculer le TF de chaque mot dans la requête
    2. Multiplier par l'IDF (déjà calculé)
    3. On obtient le vecteur TF-IDF de la query!
    """)

    # Calculer vecteur query
    query_vector = np.zeros(len(mini_engine.vocabulary))
    query_word_count = {}
    for word in query_tokens:
        if word in mini_engine.vocabulary:
            query_word_count[word] = query_word_count.get(word, 0) + 1

    query_tfidf_data = []
    for word in query_words_in_vocab:
        word_idx = mini_engine.word_to_idx[word]  # FIX: Utiliser word_to_idx!
        tf_query = query_word_count.get(word, 0) / len(query_tokens)
        idf = mini_engine.idf_vector[word_idx]
        tfidf_query = tf_query * idf
        query_vector[word_idx] = tfidf_query

        query_tfidf_data.append(
            {
                "Mot": word,
                "TF (requête)": f"{tf_query:.4f}",
                "IDF": f"{idf:.4f}",
                "TF-IDF": f"{tfidf_query:.4f}",
            }
        )

    df_query = pd.DataFrame(query_tfidf_data)
    st.markdown("**📊 Vecteur TF-IDF de ta requête:**")
    st.dataframe(df_query, use_container_width=True, hide_index=True)

    # === ÉTAPE 6: SIMILARITÉ COSINUS ===
    st.markdown("---")
    st.markdown("## 📐 Étape 6: Calcul de la Similarité Cosinus")

    st.markdown("""
    **Comment comparer la requête avec chaque document?**

    On utilise la **similarité cosinus** = mesure l'angle entre deux vecteurs!

    **Formule:**
    """)

    st.latex(r"\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}")

    st.markdown("""
    **Où:**
    - `A · B` = **Produit scalaire** (dot product)
    - `||A||` = **Norme** du vecteur A (longueur)
    - `||B||` = **Norme** du vecteur B

    **Résultat:** Score entre 0 et 1:
    - **1.0** = vecteurs identiques (angle = 0°) → **documents très similaires!** 🎯
    - **0.5** = vecteurs moyennement similaires
    - **0.0** = vecteurs orthogonaux (aucun mot en commun)
    """)

    # Calculer similarités
    results = mini_engine.search(query, top_k=len(sample_indices))

    similarity_data = []
    for doc_idx, score in results:
        doc_vector = mini_engine.tfidf_matrix[doc_idx, :]

        # Calculs détaillés
        dot_product = np.dot(query_vector, doc_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_doc = np.linalg.norm(doc_vector)

        similarity_data.append(
            {
                "Rang": len(similarity_data) + 1,
                "Document": documents_titles[doc_idx][:40] + "...",
                "Dot Product": f"{dot_product:.4f}",
                "Norme Query": f"{norm_query:.4f}",
                "Norme Doc": f"{norm_doc:.4f}",
                "Similarité": f"{score:.4f}",
                "Pertinence": "🥇 Excellent!"
                if score > 0.3
                else "🥈 Bon"
                if score > 0.1
                else "🥉 Faible",
            }
        )

    df_sim = pd.DataFrame(similarity_data)
    st.markdown("**📊 Calculs de Similarité pour Chaque Document:**")
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

    # === RÉSULTAT FINAL ===
    st.markdown("---")
    st.markdown("## 🏆 Résultat Final: Classement")

    st.markdown("""
    Les documents sont **classés par ordre décroissant** de similarité cosinus!

    Le document avec le score le plus élevé est le **plus pertinent** pour ta requête! 🎯
    """)

    # Afficher le classement final avec style
    for rank, (doc_idx, score) in enumerate(results, 1):
        if rank == 1:
            st.success(
                f"🥇 **#{rank}:** {documents_titles[doc_idx]} - Score: **{score:.4f}**"
            )
        elif rank == 2:
            st.info(
                f"🥈 **#{rank}:** {documents_titles[doc_idx]} - Score: **{score:.4f}**"
            )
        else:
            st.warning(
                f"🥉 **#{rank}:** {documents_titles[doc_idx]} - Score: **{score:.4f}**"
            )

    st.markdown("---")

    st.success("""
    ✅ **Félicitations!** Tu as vu TOUS les calculs de TF-IDF en détail!

    **Récap:**
    1. ✅ Vocabulaire construit
    2. ✅ TF calculés (fréquence locale)
    3. ✅ IDF calculés (rareté globale)
    4. ✅ TF-IDF = TF × IDF
    5. ✅ Query vectorisée
    6. ✅ Similarité cosinus calculée
    7. ✅ Documents classés!

    **🎓 Tu maîtrises maintenant TF-IDF!**
    """)


def render_tfidf_performance(
    engine, documents_texts, load_time, fit_time, remove_stopwords
):
    """Performances TF-IDF avec benchmarks automatiques et pédagogie"""
    st.header("⚡ Analyse des Performances TF-IDF")

    st.markdown("""
    Cette section t'explique **comment TF-IDF performe** et **pourquoi**!

    Tu verras:
    - Les métriques de ton corpus actuel
    - La complexité algorithmique expliquée
    - Des benchmarks automatiques sur différents datasets
    - L'impact de la taille du corpus sur la vitesse
    """)

    # ============================================================================
    # MÉTRIQUES DU CORPUS ACTUEL
    # ============================================================================
    st.markdown("### 📊 Métriques du Corpus Actuel")

    n_docs = len(documents_texts)
    n_vocab = len(engine.vocabulary)
    avg_doc_len = np.mean([len(doc) for doc in engine.documents])
    total_words = sum(len(doc) for doc in engine.documents)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Documents", f"{n_docs:,}", help="Nombre de documents indexés")
    col2.metric(
        "🔤 Vocabulaire",
        f"{n_vocab:,}",
        help="Nombre de mots uniques (après preprocessing)",
    )
    col3.metric(
        "📝 Mots/Doc", f"{avg_doc_len:.0f}", help="Longueur moyenne d'un document"
    )
    col4.metric(
        "💾 Total Mots", f"{total_words:,}", help="Nombre total de mots dans le corpus"
    )

    st.divider()

    # ============================================================================
    # TEMPS D'EXÉCUTION
    # ============================================================================
    st.markdown("### ⏱️ Temps d'Exécution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔄 Chargement**")
        st.metric("", f"{load_time:.3f}s")
        st.caption("Temps de lecture et prétraitement des données")

    with col2:
        st.markdown("**🧮 Indexation**")
        st.metric("", f"{fit_time:.3f}s")
        st.caption("Calcul des matrices TF, IDF et TF-IDF")

    with col3:
        st.markdown("**💡 Efficacité**")
        docs_per_sec = (
            n_docs / (load_time + fit_time) if (load_time + fit_time) > 0 else 0
        )
        st.metric("", f"{docs_per_sec:.0f} docs/s")
        st.caption("Nombre de documents indexés par seconde")

    # Interprétation automatique
    total_time = load_time + fit_time
    if total_time < 0.1:
        st.success(
            f"🚀 **Ultra rapide!** Indexation en {total_time:.3f}s - parfait pour ce corpus!"
        )
    elif total_time < 1.0:
        st.info(f"⚡ **Rapide!** Indexation en {total_time:.3f}s - très bon!")
    elif total_time < 5.0:
        st.info(f"👌 **Correct!** Indexation en {total_time:.3f}s - acceptable.")
    else:
        st.warning(
            f"🐌 **Lent...** Indexation en {total_time:.3f}s - corpus volumineux!"
        )

    st.divider()

    # ============================================================================
    # COMPLEXITÉ ALGORITHMIQUE
    # ============================================================================
    st.markdown("### 🧮 Complexité Algorithmique Expliquée")

    st.markdown("""
    **TF-IDF a une complexité algorithmique en `O(n × m)` où:**
    - **n** = nombre de documents
    - **m** = longueur moyenne des documents

    **Ce que ça signifie:**
    - Si tu **doubles le nombre de documents**, le temps d'indexation **double** aussi ⏱️
    - Si tu **doubles la longueur des documents**, le temps **double** aussi ⏱️

    **Opérations principales:**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Preprocessing (O(n × m)):**
        - Tokenization
        - Lowercasing
        - Stopwords removal
        - Vocabulaire construction

        **2. Calcul TF (O(n × m)):**
        - Compter occurrences
        - Normaliser par longueur
        - Stocker dans matrice
        """)

    with col2:
        st.markdown("""
        **3. Calcul IDF (O(n × v)):**
        - Compter docs contenant chaque mot
        - Appliquer log
        - Stocker dans vecteur

        **4. Calcul TF-IDF (O(n × v)):**
        - Multiplication TF × IDF
        - Stocker matrice finale
        """)

    # Estimation théorique
    st.info(f"""
    **💡 Estimation pour ton corpus:**
    - Complexité preprocessing: O({n_docs} × {avg_doc_len:.0f}) ≈ {n_docs * avg_doc_len:.0f} opérations
    - Complexité TF-IDF: O({n_docs} × {n_vocab}) ≈ {n_docs * n_vocab:.0f} opérations

    **Total estimé:** ~{(n_docs * avg_doc_len + n_docs * n_vocab):.0f} opérations
    """)

    st.divider()

    # ============================================================================
    # BENCHMARKS AUTOMATIQUES
    # ============================================================================
    st.markdown("### 🏁 Benchmarks Automatiques")

    st.markdown("""
    **On va comparer les performances** sur différents datasets pour voir l'impact de la taille! 📊

    Clique sur le bouton ci-dessous pour lancer les benchmarks (ça prend ~10-20 secondes).
    """)

    if st.button("🚀 Lancer les Benchmarks!", type="primary", key="tfidf_bench_btn"):
        with st.spinner("⏱️ Benchmarking en cours... (peut prendre 10-20s)"):
            from src.data_loader import load_dataset
            import time

            # Définir les tests
            benchmark_tests = [
                {"name": "recettes", "extended": False, "label": "Recettes (30 docs)"},
                {"name": "films", "extended": False, "label": "Films (25 docs)"},
                {
                    "name": "wikipedia",
                    "extended": False,
                    "label": "Wikipedia (50 docs)",
                },
            ]

            results = []

            for test in benchmark_tests:
                try:
                    # Charger dataset
                    start = time.time()
                    dataset = load_dataset(test["name"], extended=test["extended"])
                    texts = [doc["text"] for doc in dataset]
                    load_t = time.time() - start

                    # Indexer
                    start = time.time()
                    test_engine = TFIDFEngine(texts, remove_stopwords=remove_stopwords)
                    test_engine.fit()
                    fit_t = time.time() - start

                    # Recherche (query simple)
                    start = time.time()
                    test_engine.search("test recherche exemple", top_k=5)
                    search_t = time.time() - start

                    results.append(
                        {
                            "Dataset": test["label"],
                            "Docs": len(texts),
                            "Vocab": len(test_engine.vocabulary),
                            "Load (s)": f"{load_t:.3f}",
                            "Index (s)": f"{fit_t:.3f}",
                            "Search (s)": f"{search_t:.3f}",
                            "Total (s)": f"{load_t + fit_t:.3f}",
                            "_total_numeric": load_t + fit_t,
                            "_docs_numeric": len(texts),
                        }
                    )
                except Exception as e:
                    st.error(f"Erreur sur {test['label']}: {e}")
                    continue

            if results:
                # Afficher tableau
                df_results = pd.DataFrame(results)
                df_display = df_results.drop(
                    columns=["_total_numeric", "_docs_numeric"]
                )

                st.markdown("**📊 Résultats des Benchmarks:**")
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Graphique: Temps vs Nombre de docs
                st.markdown(
                    "**📈 Graphique: Temps d'Indexation vs Nombre de Documents**"
                )

                col_graph, col_analysis = st.columns([2, 1])

                with col_graph:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 5))

                    x = [r["_docs_numeric"] for r in results]
                    y = [r["_total_numeric"] for r in results]
                    labels = [r["Dataset"] for r in results]

                    # Scatter plot
                    ax.scatter(x, y, s=100, alpha=0.6, color="#1f77b4")

                    # Labels
                    for i, label in enumerate(labels):
                        ax.annotate(
                            label,
                            (x[i], y[i]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                        )

                    # Ligne de tendance
                    if len(x) > 1:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), "r--", alpha=0.8, label="Tendance linéaire")
                        ax.legend()

                    ax.set_xlabel("Nombre de Documents")
                    ax.set_ylabel("Temps Total (s)")
                    ax.set_title("Performance TF-IDF: Temps vs Taille du Corpus")
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                with col_analysis:
                    st.markdown("**🔍 Analyse:**")

                    fastest = min(results, key=lambda x: x["_total_numeric"])
                    slowest = max(results, key=lambda x: x["_total_numeric"])

                    st.markdown(f"""
                    **⚡ Plus rapide:**
                    {fastest["Dataset"]}
                    - {fastest["Total (s)"]}s
                    - {fastest["Docs"]} docs

                    **🐌 Plus lent:**
                    {slowest["Dataset"]}
                    - {slowest["Total (s)"]}s
                    - {slowest["Docs"]} docs

                    **💡 Observation:**

                    La ligne rouge montre la tendance **linéaire** → confirme la complexité O(n)!

                    Plus il y a de documents, plus ça prend de temps **proportionnellement**.
                    """)

                st.success("""
                ✅ **Conclusion des Benchmarks:**

                TF-IDF est **rapide et scalable** pour des corpus de taille petite à moyenne!

                - **< 100 docs:** Quasi instantané ⚡
                - **100-1000 docs:** Très rapide (< 1s) 🚀
                - **1000-10000 docs:** Rapide (1-10s) 👌
                - **> 10000 docs:** Optimisations recommandées (index inversé, etc.)
                """)

    st.divider()

    # ============================================================================
    # OPTIMISATIONS POSSIBLES
    # ============================================================================
    st.markdown("### 🚀 Optimisations Possibles")

    st.markdown("""
    Si ton corpus devient **très gros** (> 10,000 docs), voici comment accélérer TF-IDF:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Index Inversé**

        Au lieu de stocker une matrice complète (docs × mots), stocke seulement les mots **présents** dans chaque document.

        ➡️ Économise de la RAM et accélère la recherche!

        **2. Sparse Matrices**

        Utilise `scipy.sparse` au lieu de NumPy dense.

        ➡️ Matrice TF-IDF souvent > 90% de zéros!

        **3. Preprocessing Cache**

        Sauvegarde les documents preprocessés sur disque.

        ➡️ Évite de retokenizer à chaque run!
        """)

    with col2:
        st.markdown("""
        **4. Batch Processing**

        Traite les documents par batch de 1000.

        ➡️ Évite les pics de RAM!

        **5. Parallelization**

        Utilise `multiprocessing` pour tokenizer en parallèle.

        ➡️ CPU multi-core = gain de vitesse!

        **6. Approximations**

        Limite le vocabulaire aux N mots les plus fréquents.

        ➡️ Trade-off précision vs vitesse!
        """)

    st.info("""
    **💡 Pour ton usage actuel:**

    Avec des corpus de ~1000 docs, **aucune optimisation n'est nécessaire**!

    TF-IDF est déjà **rapide** et **efficace** pour cette taille. 🎯
    """)


# ============================================================================
# SECTION BM25 (NOUVEAU!)
# ============================================================================


def render_bm25_section(
    dataset,
    documents_texts,
    documents_titles,
    documents_categories,
    tfidf_engine,
    remove_stopwords,
):
    """Section BM25 complète"""

    st.title("🎯 BM25: Best Matching 25 - TF-IDF Amélioré")

    # Sub-navigation avec boutons stylés
    tabs_bm25 = [
        "📖 Introduction",
        "🔢 Concepts",
        "🔍 Recherche",
        "📊 Exploration",
        "🎓 Pas-à-Pas",
        "⚔️ Comparaison",
        "⚡ Performance",
    ]
    tab = render_tab_navigation(tabs_bm25, "bm25_current_tab")

    if tab == "📖 Introduction":
        render_bm25_intro()
    elif tab == "🔢 Concepts":
        render_bm25_concepts(documents_texts, remove_stopwords)
    elif tab == "🔍 Recherche":
        render_bm25_search(
            documents_texts, documents_titles, documents_categories, remove_stopwords
        )
    elif tab == "📊 Exploration":
        render_bm25_exploration(documents_texts, documents_titles, remove_stopwords)
    elif tab == "🎓 Pas-à-Pas":
        render_bm25_stepbystep(documents_texts, documents_titles, remove_stopwords)
    elif tab == "⚔️ Comparaison":
        render_bm25_comparison(
            documents_texts, documents_titles, tfidf_engine, remove_stopwords
        )
    elif tab == "⚡ Performance":
        render_bm25_performance(documents_texts, remove_stopwords)


def render_bm25_intro():
    """Introduction BM25 & Problèmes de TF-IDF - HYPER PÉDAGOGIQUE"""
    st.header("📖 BM25: Évolution Intelligente de TF-IDF")

    st.markdown("""
    ### 🎯 Contexte Historique

    **TF-IDF** (années 1970) était révolutionnaire pour son époque.
    **BM25** (1994) est son évolution moderne, développée par Stephen Robertson et Karen Spärck Jones.

    BM25 = **B**est **M**atching **25** (25ème itération de l'algorithme!)
    """)

    st.divider()

    st.markdown("## ❌ Les 3 Problèmes Fondamentaux de TF-IDF")

    # === PROBLÈME #1: SATURATION ===
    with st.expander("**🔴 Problème #1: Saturation Linéaire du TF**", expanded=True):
        st.markdown("""
        ### 💡 Le Problème en Détail

        Dans TF-IDF, le score croît **linéairement** avec le nombre d'occurrences.
        Mais est-ce réaliste? 🤔
        """)

        # Exemple concret avec calculs
        st.markdown("""
        #### 📝 Exemple Concret

        Imaginons 3 documents parlant de "Python":
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("""
            **📄 Doc A**

            - Longueur: 100 mots
            - "python": **1× fois**
            - TF = 1/100 = **0.01**

            *Article de blog mentionnant Python en passant*
            """)

        with col2:
            st.warning("""
            **📄 Doc B**

            - Longueur: 100 mots
            - "python": **10× fois**
            - TF = 10/100 = **0.10**

            *Article dédié à Python*
            """)

        with col3:
            st.error("""
            **📄 Doc C**

            - Longueur: 100 mots
            - "python": **50× fois**
            - TF = 50/100 = **0.50**

            *Spam avec répétitions*
            """)

        st.markdown("""
        ### 🤯 Le Problème

        Avec TF-IDF:
        - Doc B (10×) a un score **10× plus élevé** que Doc A (1×)
        - Doc C (50×) a un score **5× plus élevé** que Doc B (10×)

        **Mais en réalité:**
        - Après 10 occurrences, le mot n'apporte **plus d'information nouvelle**!
        - Doc C n'est pas 50× plus pertinent que Doc A
        - On veut un **effet de saturation** (plateau)
        """)

        # Graphique en colonnes
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_sat = plot_saturation_effect()
            st.pyplot(fig_sat)

        with col_g2:
            st.markdown("""
            ### 📊 Analyse du Graphique

            **Ligne rouge (TF-IDF):**
            - Croissance linéaire infinie
            - 100 occurrences = 100× le score de 1
            - **Irréaliste!** ❌

            **Courbes colorées (BM25):**
            - Croissance rapide au début
            - Plateau après N occurrences
            - **Réaliste!** ✅

            **Paramètre k1:**
            - k1 faible → saturation rapide
            - k1 élevé → saturation lente
            """)

    # === PROBLÈME #2: NORMALISATION ===
    with st.expander("**🟠 Problème #2: Normalisation Naïve par Longueur**"):
        st.markdown("""
        ### 💡 Le Problème en Détail

        TF-IDF normalise en divisant simplement par la longueur totale.
        Résultat: les documents longs sont **toujours pénalisés**.
        """)

        st.markdown("""
        #### 📝 Exemple Concret

        Deux recettes parlant de "chocolat":
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **🍫 Recette Courte (50 mots)**

            ```
            Mousse au chocolat simple:
            200g chocolat, 4 oeufs, sucre.
            Faire fondre chocolat.
            Monter blancs. Mélanger.
            [+ 42 mots de remplissage...]
            ```

            - "chocolat": **2 occurrences**
            - TF = 2/50 = **0.04** (4%)
            """)

        with col2:
            st.warning("""
            **🍫 Recette Détaillée (500 mots)**

            ```
            Mousse au chocolat professionnelle:
            Introduction sur le chocolat (50 mots)
            Liste détaillée ingrédients (100 mots)
            Étapes détaillées avec chocolat (200 mots)
            Astuces et variantes chocolat (150 mots)
            ```

            - "chocolat": **15 occurrences**
            - TF = 15/500 = **0.03** (3%)
            """)

        st.markdown("""
        ### 🤯 Le Problème

        **Avec TF-IDF:**
        - Recette courte (2× chocolat) → TF = 0.04
        - Recette détaillée (15× chocolat) → TF = 0.03
        - La recette détaillée a un **score PLUS BAS** ❌

        **Ce qu'on veut:**
        - Pénaliser les docs longs... **mais pas toujours!**
        - Certains corpus ont naturellement des docs longs (articles scientifiques)
        - D'autres ont des docs courts (tweets, recettes)
        - On veut un **contrôle ajustable** via paramètre **b**
        """)

        st.markdown("""
        ### 💡 Solution BM25

        Le paramètre **b** contrôle l'intensité de la pénalité:

        - **b = 0**: Aucune pénalité (ignore la longueur)
        - **b = 0.5**: Pénalité légère
        - **b = 0.75**: Pénalité standard ⭐ (recommandé)
        - **b = 1.0**: Pénalité complète (comme TF-IDF)

        Tu peux adapter selon ton corpus!
        """)

    # === PROBLÈME #3: PAS DE CONTRÔLE ===
    with st.expander("**🟡 Problème #3: Aucun Paramètre Ajustable**"):
        st.markdown("""
        ### 💡 Le Problème

        TF-IDF est une formule **figée**:
        """)

        st.latex(
            r"\text{TF-IDF} = \frac{f}{|D|} \times \log\left(\frac{N}{n(t)}\right)"
        )

        st.markdown("""
        **Conséquences:**
        - ❌ Impossible d'adapter selon le type de corpus
        - ❌ Impossible de tuner pour de meilleures performances
        - ❌ Un seul comportement pour tous les cas

        **Exemples de corpus différents:**

        | Type de Corpus | Comportement Optimal |
        |----------------|---------------------|
        | **Tweets** (courts) | Peu de normalisation (b faible) |
        | **Articles** (longs) | Normalisation forte (b élevé) |
        | **Spam** (répétitions) | Saturation rapide (k1 faible) |
        | **Littérature** (varié) | Saturation lente (k1 élevé) |

        TF-IDF ne peut s'adapter à aucun de ces cas! ❌
        """)

    st.divider()

    # === SOLUTION BM25 ===
    st.markdown("## ✅ BM25: La Solution Intelligente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        ### 🎛️ Paramètre k1

        **Contrôle la saturation du TF**

        **Valeurs typiques:**
        - 0.5 → Saturation rapide
        - **1.5** → Standard ⭐
        - 2.0 → Saturation lente
        - ∞ → Comme TF-IDF

        **Effet:**
        Après N occurrences, le score plafonne intelligemment!
        """)

    with col2:
        st.success("""
        ### ⚖️ Paramètre b

        **Contrôle la normalisation**

        **Valeurs typiques:**
        - 0.0 → Aucune
        - **0.75** → Standard ⭐
        - 1.0 → Complète

        **Effet:**
        Ajuste la pénalité des documents longs selon ton corpus!
        """)

    with col3:
        st.success("""
        ### 📊 IDF Amélioré

        **Smoothing intégré**

        **Formule:**
        ```
        log((N - n + 0.5) /
            (n + 0.5))
        ```

        **Effet:**
        - Évite divisions par zéro
        - Plus stable que TF-IDF
        - Meilleure gestion des mots rares
        """)

    st.divider()

    st.markdown("## 🎯 Formule Complète BM25")

    st.latex(r"""
    \text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k1 + 1)}{f(q_i, D) + k1 \times \left(1 - b + b \times \frac{|D|}{\text{avgdl}}\right)}
    """)

    st.markdown(r"""
    ### 📖 Décomposition de la Formule

    | Composant | Signification | Rôle |
    |-----------|---------------|------|
    | **IDF(qi)** | Inverse Document Frequency | Rareté du mot (avec smoothing) |
    | **f(qi, D)** | Fréquence du mot | Nombre d'occurrences dans le doc |
    | **k1** | Paramètre saturation | Contrôle le plateau du TF |
    | **b** | Paramètre normalisation | Contrôle la pénalité de longueur |
    | **\|D\|** | Longueur du document | Nombre de mots dans le doc |
    | **avgdl** | Longueur moyenne | Moyenne du corpus |

    ### 💡 En Résumé

    **TF-IDF:** Simple mais limité (années 1970)
    **BM25:** Intelligent et ajustable (1994 - encore utilisé aujourd'hui!)

    BM25 est le **standard industriel** pour la recherche textuelle:
    - Utilisé par Elasticsearch
    - Utilisé par Apache Lucene/Solr
    - Base de millions de moteurs de recherche
    """)


def render_bm25_concepts(documents_texts, remove_stopwords):
    """Concepts BM25 détaillés - HYPER PÉDAGOGIQUE"""
    st.header("🔢 Comprendre BM25 en Profondeur")

    st.markdown("""
    Décortiquons chaque composant de BM25 avec des **exemples concrets** et des **calculs chiffrés**!
    """)

    # === IDF AMÉLIORÉ ===
    with st.expander(
        "📉 **Composant #1: IDF Amélioré (avec smoothing)**", expanded=True
    ):
        st.markdown("""
        ### 💡 L'IDF de TF-IDF avait des Problèmes

        **Formule TF-IDF IDF:**
        """)

        st.latex(r"\text{IDF}_{\text{TF-IDF}}(t) = \log\left(\frac{N}{n(t)}\right)")

        st.markdown("""
        **Problèmes:**
        1. ⚠️ **Division par zéro** si n(t) = 0 (mot absent du corpus)
        2. ⚠️ **Valeurs extrêmes** pour les mots très rares
        3. ⚠️ **Pas de smoothing** pour stabiliser
        """)

        st.divider()

        st.markdown("""
        ### ✅ BM25 IDF Résout Ces Problèmes

        **Formule BM25 IDF:**
        """)

        st.latex(
            r"\text{IDF}_{\text{BM25}}(q) = \log\left(\frac{N - n(q) + 0.5}{n(q) + 0.5}\right)"
        )

        st.markdown("""
        **Composants:**
        - **N** = nombre total de documents dans le corpus
        - **n(q)** = nombre de documents contenant le terme q
        - **+0.5** = **smoothing de Laplace** (évite divisions par zéro)
        """)

        st.markdown("""
        ### 📝 Exemple Concret avec Calculs

        Corpus de **1000 documents**:
        """)

        # Tableau comparatif avec calculs
        import pandas as pd

        examples = [
            {"Mot": "le", "n(q)": 950, "Rareté": "Très commun"},
            {"Mot": "cuisine", "n(q)": 300, "Rareté": "Commun"},
            {"Mot": "python", "n(q)": 50, "Rareté": "Rare"},
            {"Mot": "blockchain", "n(q)": 5, "Rareté": "Très rare"},
        ]

        for ex in examples:
            n = ex["n(q)"]
            N = 1000
            # TF-IDF IDF
            idf_tfidf = np.log(N / n) if n > 0 else float("inf")
            # BM25 IDF
            idf_bm25 = np.log((N - n + 0.5) / (n + 0.5))

            ex["IDF TF-IDF"] = f"{idf_tfidf:.3f}"
            ex["IDF BM25"] = f"{idf_bm25:.3f}"

        df_idf = pd.DataFrame(examples)
        st.dataframe(df_idf, use_container_width=True)

        st.markdown("""
        ### 📊 Observations

        **Pour les mots communs ("le"):**
        - TF-IDF: 0.054 (très proche de 0)
        - BM25: 0.053 (similaire)
        - ✅ Peu de différence

        **Pour les mots rares ("blockchain"):**
        - TF-IDF: 5.298 (valeur élevée)
        - BM25: 5.298 (plus stable avec smoothing)
        - ✅ BM25 mieux stabilisé

        **Avantage du +0.5:**
        - Évite les explosions de valeurs
        - Plus robuste aux mots très rares
        - Meilleure généralisation
        """)

    # === SATURATION DU TF ===
    with st.expander("🎛️ **Composant #2: Saturation du TF (Paramètre k1)**"):
        st.markdown("""
        ### 💡 Pourquoi Saturer le TF?

        **Observation réaliste:**
        Après **10 occurrences**, un mot n'apporte **plus beaucoup d'info nouvelle**.

        - 1 occurrence → Le doc parle du sujet ✅
        - 10 occurrences → Le doc parle VRAIMENT du sujet ✅✅
        - 100 occurrences → Le doc... parle toujours du sujet (mais pas 100× plus!) ⚠️
        """)

        st.markdown("""
        ### 🔢 Formule du TF Saturé
        """)

        st.latex(r"\text{TF}_{\text{BM25}} = \frac{f \times (k1 + 1)}{f + k1}")

        st.markdown("""
        **Composants:**
        - **f** = fréquence du terme dans le document
        - **k1** = paramètre contrôlant la vitesse de saturation
        """)

        st.markdown("""
        ### 📝 Exemple Concret: Mot "Python"

        Testons différentes valeurs de **k1**:
        """)

        # Calculs pour différents k1
        frequencies = [1, 2, 5, 10, 20, 50, 100]
        k1_values = [0.5, 1.2, 1.5, 2.0]

        data = []
        for f in frequencies:
            row = {"Occurrences (f)": f}
            for k1 in k1_values:
                tf_bm25 = (f * (k1 + 1)) / (f + k1)
                row[f"k1={k1}"] = f"{tf_bm25:.3f}"
            data.append(row)

        df_saturation = pd.DataFrame(data)
        st.dataframe(df_saturation, use_container_width=True)

        st.markdown("""
        ### 📊 Observations Clés

        **Avec k1 = 0.5 (saturation rapide):**
        - 10 occ → 0.909
        - 100 occ → 0.993
        - Plafonne très vite! (bon pour éviter le spam)

        **Avec k1 = 1.5 (standard ⭐):**
        - 10 occ → 1.304
        - 100 occ → 1.485
        - Équilibre réaliste

        **Avec k1 = 2.0 (saturation lente):**
        - 10 occ → 1.375
        - 100 occ → 1.970
        - Plus de croissance (bon pour textes variés)

        **k1 → ∞ (comme TF-IDF):**
        - Croissance linéaire sans limite
        """)

        # Graphique en colonnes
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_sat = plot_saturation_effect(k1_values=[0.5, 1.2, 1.5, 2.0])
            st.pyplot(fig_sat)

        with col_g2:
            st.markdown("""
            ### 📈 Analyse du Graphique

            **Axe X:** Nombre d'occurrences du mot
            **Axe Y:** Score TF résultant

            **Ligne rouge (TF-IDF):**
            - Monte indéfiniment
            - 100 occ = 100× le score
            - **Problématique!** ❌

            **Courbes BM25:**
            - **Bleue (k1=0.5)**: Plateau à ~1.0
            - **Orange (k1=1.2)**: Plateau à ~1.2
            - **Verte (k1=1.5)**: Plateau à ~1.5 ⭐
            - **Rouge (k1=2.0)**: Plateau à ~2.0

            **Conseil:**
            k1=1.5 est le standard pour la plupart des corpus!
            """)

    # === NORMALISATION ===
    with st.expander("⚖️ **Composant #3: Normalisation de Longueur (Paramètre b)**"):
        st.markdown("""
        ### 💡 Le Problème des Documents Longs

        **Question:** Un document de 1000 mots devrait-il être pénalisé
        par rapport à un document de 100 mots?

        **Réponse:** **Ça dépend du corpus!** 🤔

        - **Tweets** (courts naturellement) → Peu de pénalité
        - **Articles scientifiques** (longs naturellement) → Forte pénalité
        - **Recettes** (longueur mixte) → Pénalité modérée
        """)

        st.markdown("""
        ### 🔢 Formule de Normalisation
        """)

        st.latex(r"\text{norm} = 1 - b + b \times \frac{|D|}{\text{avgdl}}")

        st.markdown("""
        **Composants:**
        - **|D|** = longueur du document actuel (en mots)
        - **avgdl** = longueur moyenne du corpus (average document length)
        - **b** = intensité de la pénalité (0 = aucune, 1 = complète)
        """)

        st.markdown("""
        ### 📝 Exemple Concret

        Corpus avec **avgdl = 100 mots** (moyenne):
        """)

        # Calculs pour différents b
        doc_lengths = [50, 100, 200, 500, 1000]
        b_values = [0.0, 0.5, 0.75, 1.0]
        avgdl = 100

        data = []
        for length in doc_lengths:
            row = {"Longueur doc": f"{length} mots"}
            for b in b_values:
                norm = 1 - b + b * (length / avgdl)
                row[f"b={b}"] = f"{norm:.3f}"
            data.append(row)

        df_norm = pd.DataFrame(data)
        st.dataframe(df_norm, use_container_width=True)

        st.markdown("""
        ### 📊 Interprétation

        **Facteur de normalisation > 1** = Pénalité (doc plus long que la moyenne)
        **Facteur de normalisation = 1** = Neutre (doc de longueur moyenne)
        **Facteur de normalisation < 1** = Boost (doc plus court que la moyenne)

        **Avec b = 0 (aucune normalisation):**
        - Facteur = 1.0 pour tous les docs
        - La longueur est **ignorée**
        - Bon pour corpus homogènes

        **Avec b = 0.75 (standard ⭐):**
        - Doc 50 mots → 0.625 (boost de +60%)
        - Doc 200 mots → 1.750 (pénalité de -43%)
        - Doc 1000 mots → 8.500 (pénalité de -88%)
        - Équilibre raisonnable

        **Avec b = 1.0 (normalisation complète):**
        - Pénalité maximale pour les docs longs
        - Comme TF-IDF (division par longueur)
        """)

        # Créer un mini corpus pour calculer avgdl
        bm25_demo = BM25Engine(documents_texts[:10], remove_stopwords=remove_stopwords)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_norm = plot_length_normalization(
                avgdl=bm25_demo.avgdl, doc_lengths=[50, 100, 150, 200]
            )
            st.pyplot(fig_norm)

        with col_g2:
            st.markdown(f"""
            ### 📈 Analyse du Graphique

            **Corpus actuel:**
            - avgdl = {bm25_demo.avgdl:.1f} mots

            **Ligne horizontale (b=0):**
            - Facteur = 1.0 constant
            - Aucune pénalité

            **Courbes montantes (b > 0):**
            - Plus b est élevé, plus la pente est forte
            - b=0.75 (standard) = compromis

            **Conseil pratique:**
            - **b=0.5** si corpus homogène (longueurs similaires)
            - **b=0.75** standard (recommandé) ⭐
            - **b=1.0** si beaucoup de spam/docs longs
            """)

    # === FORMULE COMPLÈTE ===
    with st.expander("🎯 **Formule Complète BM25**"):
        st.markdown("""
        ### 🔥 La Grande Formule (tout ensemble!)
        """)

        st.latex(r"""
        \text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k1 + 1)}{f(q_i, D) + k1 \times \left(1 - b + b \times \frac{|D|}{\text{avgdl}}\right)}
        """)

        st.markdown("""
        ### 📖 Décortiquons la Formule

        **Σ (Somme):** On additionne pour **chaque mot** de la query

        **IDF(qi):** Rareté du mot i dans le corpus (avec smoothing)

        **Numérateur:** f(qi, D) × (k1 + 1)
        → Fréquence du mot, légèrement boostée

        **Dénominateur:** f(qi, D) + k1 × [1 - b + b × |D|/avgdl]
        → Fréquence + facteur de saturation × normalisation
        """)

        st.markdown("""
        ### 🎓 Algorithme en Pseudo-Code

        ```python
        def BM25(document, query, k1=1.5, b=0.75):
            score = 0

            for mot in query:
                # 1. Calculer IDF
                idf = log((N - n + 0.5) / (n + 0.5))

                # 2. Compter fréquence
                f = count(mot, document)

                # 3. Calculer normalisation
                norm = 1 - b + b * (len(document) / avgdl)

                # 4. Calculer TF saturé
                tf = (f * (k1 + 1)) / (f + k1 * norm)

                # 5. Multiplier IDF × TF
                score += idf * tf

            return score
        ```
        """)

        st.success("""
        ### ✅ Avantages de BM25

        1. **Saturation intelligente** via k1 (évite le spam)
        2. **Normalisation ajustable** via b (adapte au corpus)
        3. **IDF stable** avec smoothing (robuste)
        4. **Paramètres tunables** (optimisation possible)
        5. **Standard industriel** (utilisé partout!)
        """)


def render_bm25_search(
    documents_texts, documents_titles, documents_categories, remove_stopwords
):
    """Recherche interactive BM25"""
    st.header("🔍 Recherche Interactive BM25")

    st.markdown("""
    Teste BM25 avec tes propres paramètres!
    """)

    # Utiliser un formulaire pour soumission avec Enter
    with st.form("bm25_search_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            query = st.text_input(
                "🔎 Votre recherche:",
                value="plat italien fromage",  # Valeur par défaut!
                placeholder="Ex: plat italien, film science-fiction, guerre mondiale...",
                key="bm25_query_input",
                help='💡 **Exemples:** "plat italien fromage" | "science-fiction vaisseau espace" | "guerre conflit mondial armée" | "football champion coupe"',
            )

        with col2:
            k1 = st.slider(
                "k1 (saturation)",
                min_value=0.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="⚙️ Contrôle la saturation du TF. **Standard = 1.5** | Plus élevé = moins de saturation | Plus bas = saturation rapide",
                key="bm25_k1_slider",
            )

        with col3:
            b = st.slider(
                "b (normalisation)",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="⚙️ Contrôle la pénalité de longueur. **Standard = 0.75** | 0 = aucune | 1 = complète",
                key="bm25_b_slider",
            )

        top_k = st.slider(
            "Nombre de résultats:",
            3,
            20,
            5,
            key="bm25_topk_slider",
            help="Nombre de documents les plus pertinents à afficher",
        )

        # Bouton de soumission (Enter fonctionne aussi!)
        submitted = st.form_submit_button("🚀 Rechercher avec BM25!", type="primary")

    if submitted and query:
        with st.spinner("🔍 Recherche BM25 en cours..."):
            # Créer engine BM25 avec les paramètres
            bm25_engine = BM25Engine(
                documents_texts, k1=k1, b=b, remove_stopwords=remove_stopwords
            )

            results = bm25_engine.search(query, top_k=top_k)

            if len(results) == 0 or all(score == 0 for _, score in results):
                st.warning("😕 Aucun résultat. Essaie d'autres mots!")
            else:
                st.success(f"✅ {len(results)} résultats BM25 trouvés!")

                # === ANALYSE DES PARAMÈTRES ===
                st.markdown("### ⚙️ Interprétation de tes Paramètres")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Analyse k1
                    if k1 < 1.0:
                        k1_interpretation = "**Saturation rapide** 🚀 (anti-spam)"
                        k1_color = "🟢"
                    elif k1 < 1.8:
                        k1_interpretation = "**Standard** ⭐ (équilibré)"
                        k1_color = "🟡"
                    else:
                        k1_interpretation = (
                            "**Saturation lente** 🐌 (favorise répétitions)"
                        )
                        k1_color = "🔴"

                    st.info(f"""
                    **k1 = {k1}** {k1_color}

                    {k1_interpretation}

                    Impact: Les mots plafonnent après ~{int(k1 * 10)} occurrences
                    """)

                with col2:
                    # Analyse b
                    if b < 0.5:
                        b_interpretation = (
                            "**Faible normalisation** (favorise docs longs)"
                        )
                        b_color = "🟢"
                    elif b < 0.85:
                        b_interpretation = "**Normalisation standard** ⭐"
                        b_color = "🟡"
                    else:
                        b_interpretation = (
                            "**Forte normalisation** (pénalise docs longs)"
                        )
                        b_color = "🔴"

                    st.info(f"""
                    **b = {b}** {b_color}

                    {b_interpretation}

                    avgdl = {bm25_engine.avgdl:.1f} mots
                    """)

                with col3:
                    # Stats query
                    query_words = query.lower().split()
                    st.info(f"""
                    **Query:** {len(query_words)} mots

                    Mots: {", ".join(query_words[:3])}{"..." if len(query_words) > 3 else ""}

                    Corpus: {len(documents_texts)} docs
                    """)

                st.divider()

                # === VISUALISATION RÉSULTATS ===
                st.markdown("### 📊 Visualisation des Scores")

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    fig_results = plot_search_results(results, documents_titles, query)
                    st.pyplot(fig_results)

                with col_g2:
                    st.markdown("### 📈 Analyse des Scores")

                    all_scores = [score for _, score in results]
                    max_score = max(all_scores)
                    min_score = min(all_scores)
                    avg_score = np.mean(all_scores)

                    st.markdown(f"""
                    **Statistiques:**
                    - 🥇 Score max: **{max_score:.4f}**
                    - 🥉 Score min: **{min_score:.4f}**
                    - 📊 Score moyen: **{avg_score:.4f}**
                    - 📏 Écart: **{(max_score - min_score):.4f}**

                    **Observations:**
                    """)

                    # Analyses automatiques
                    if max_score > avg_score * 3:
                        st.success(
                            "✅ **Excellente séparation!** Le meilleur résultat se démarque clairement."
                        )
                    elif max_score > avg_score * 1.5:
                        st.info(
                            "💡 **Bonne séparation.** Les résultats sont bien différenciés."
                        )
                    else:
                        st.warning(
                            "⚠️ **Faible séparation.** Les documents ont des scores similaires. Essaie de modifier k1 ou b."
                        )

                    if min_score < 0.1:
                        st.info(
                            "📉 Les derniers résultats ont des scores très faibles (< 0.1). Ils contiennent peu de mots de ta query."
                        )

                # === RÉSULTATS DÉTAILLÉS ===
                st.markdown("### 🎯 Top Résultats Détaillés")

                for rank, (doc_idx, score) in enumerate(results[:5], 1):
                    # Calcul du pourcentage par rapport au max
                    score_pct = (score / max_score * 100) if max_score > 0 else 0

                    with st.expander(
                        f"#{rank} - {documents_titles[doc_idx]} | BM25: {score:.3f} ({score_pct:.0f}%)"
                    ):
                        st.caption(f"📂 Catégorie: {documents_categories[doc_idx]}")
                        st.write(documents_texts[doc_idx][:300] + "...")

                        # Info sur la longueur du doc
                        doc_length = len(documents_texts[doc_idx].split())
                        length_ratio = doc_length / bm25_engine.avgdl

                        col_info1, col_info2, col_info3 = st.columns(3)

                        with col_info1:
                            st.metric("Longueur", f"{doc_length} mots")
                        with col_info2:
                            st.metric("vs. Moyenne", f"{length_ratio:.2f}×")
                        with col_info3:
                            if length_ratio > 1.5:
                                st.metric("Type", "Long 📜")
                            elif length_ratio < 0.7:
                                st.metric("Type", "Court 📋")
                            else:
                                st.metric("Type", "Moyen 📄")

                        if st.checkbox(
                            f"🔍 Voir calcul détaillé #{rank}",
                            key=f"bm25_explain_{rank}",
                        ):
                            explanation = bm25_engine.explain(query, doc_idx)

                            st.markdown("""
                            #### 📐 Détails du Calcul BM25
                            """)

                            # Afficher les valeurs clés
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "avgdl (corpus)", f"{explanation['avgdl']:.1f} mots"
                                )
                            with col2:
                                st.metric(
                                    "Longueur doc", f"{explanation['doc_length']} mots"
                                )
                            with col3:
                                st.metric(
                                    "Facteur norm", f"{explanation['norm_factor']:.3f}"
                                )

                            st.markdown(f"""
                            **Interprétation du facteur de normalisation:**
                            - Valeur: **{explanation["norm_factor"]:.3f}**
                            - Si > 1: Document **pénalisé** (plus long que la moyenne)
                            - Si = 1: Document de longueur **moyenne**
                            - Si < 1: Document **boosté** (plus court que la moyenne)

                            **Score final BM25:** **{explanation["total_score"]:.4f}**
                            """)


def render_bm25_exploration(documents_texts, documents_titles, remove_stopwords):
    """Exploration & Tuning BM25"""
    st.header("📊 Exploration & Tuning des Paramètres")

    st.markdown("""
    ### 🎛️ Laboratoire de Tuning BM25

    Explore l'impact des paramètres k1 et b sur les scores!
    """)

    # Sélection document
    doc_idx = st.selectbox(
        "Choisis un document:",
        range(min(20, len(documents_titles))),
        format_func=lambda x: documents_titles[x],
    )

    test_query = st.text_input(
        "Query de test:",
        value="recette cuisine",
        key="bm25_tuning_query",
        help='💡 Exemples: "plat italien" | "cuisine asiatique" | "dessert chocolat"',
    )

    if test_query:
        with st.spinner("🧪 Génération de la heatmap..."):
            bm25_engine = BM25Engine(documents_texts, remove_stopwords=remove_stopwords)

            fig_heatmap = plot_parameter_space_heatmap(
                bm25_engine,
                test_query,
                doc_idx,
                k1_range=(0.5, 3.0),
                b_range=(0.0, 1.0),
                resolution=15,
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.info("""
            💡 **Interprétation:**
            - Zones **rouges** = scores élevés
            - ⭐ **Étoile blanche** = paramètres standard (k1=1.5, b=0.75)
            - Explore l'espace pour voir l'impact!
            """)


def render_bm25_stepbystep(documents_texts, documents_titles, remove_stopwords):
    """Exemple pas-à-pas BM25 - HYPER DÉTAILLÉ"""
    st.header("🎓 Calcul BM25 Pas-à-Pas (Exemple Complet)")

    st.markdown("""
    Suivons **TOUTES les étapes** du calcul BM25, de A à Z, avec des **exemples concrets**!

    On va décortiquer chaque formule et voir comment BM25 classe les documents.
    """)

    # === SÉLECTION DOCUMENTS ===
    st.markdown("## 📄 Documents d'Exemple")

    sample_indices = list(range(min(3, len(documents_texts))))
    sample_texts = [documents_texts[i] for i in sample_indices]
    sample_titles = [documents_titles[i] for i in sample_indices]

    for idx, (title, text) in enumerate(zip(sample_titles, sample_texts)):
        with st.expander(f"📄 **Document {idx + 1}:** {title}"):
            st.write(text)
            st.caption(f"Longueur: {len(text.split())} mots")

    st.divider()

    # === QUERY ===
    query = st.text_input(
        "🔎 Entre ta Query:",
        value="italien fromage",
        key="bm25_tutorial",
        help='💡 Teste avec: "italien fromage" | "chocolat dessert" | "poisson grillé" | "asiatique épicé"',
    )

    if query:
        # === PARAMÈTRES ===
        st.markdown("## ⚙️ Paramètres BM25")

        col1, col2 = st.columns(2)
        with col1:
            k1_tutorial = st.number_input(
                "k1 (saturation):",
                min_value=0.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                key="bm25_tutorial_k1",
                help="Contrôle la saturation du TF. Standard = 1.5",
            )
        with col2:
            b_tutorial = st.number_input(
                "b (normalisation):",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                key="bm25_tutorial_b",
                help="Contrôle la pénalité de longueur. Standard = 0.75",
            )

        st.divider()

        # Créer l'engine BM25
        mini_bm25 = BM25Engine(
            sample_texts,
            k1=k1_tutorial,
            b=b_tutorial,
            remove_stopwords=remove_stopwords,
        )

        # Preprocessing de la query
        from src.preprocessing import preprocess_text

        query_words = preprocess_text(query, remove_stopwords=remove_stopwords)

        # === ÉTAPE 0: STATS CORPUS ===
        with st.expander("**📊 Étape 0: Statistiques du Mini-Corpus**", expanded=False):
            st.markdown("""
            ### Avant de calculer BM25, analysons notre corpus!
            """)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Nombre de documents (N)", mini_bm25.N)
            with col2:
                st.metric("Vocabulaire", len(mini_bm25.vocabulary))
            with col3:
                st.metric("Longueur moyenne (avgdl)", f"{mini_bm25.avgdl:.1f} mots")

            # Tableau des longueurs
            lengths_data = []
            for idx, text in enumerate(sample_texts):
                doc_length = len(text.split())
                ratio = doc_length / mini_bm25.avgdl
                lengths_data.append(
                    {
                        "Document": sample_titles[idx][:30],
                        "Longueur": doc_length,
                        "vs. Moyenne": f"{ratio:.2f}×",
                    }
                )

            df_lengths = pd.DataFrame(lengths_data)
            st.dataframe(df_lengths, use_container_width=True)

            st.info(f"""
            💡 **avgdl** est crucial pour BM25! Les documents plus longs que {mini_bm25.avgdl:.1f} mots seront pénalisés (si b > 0).
            """)

        # === ÉTAPE 1: IDF ===
        with st.expander(
            "**📉 Étape 1: Calcul des IDF (Inverse Document Frequency)**",
            expanded=False,
        ):
            st.markdown("""
            ### Formule BM25 IDF (avec smoothing)
            """)

            st.latex(
                r"\text{IDF}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5}\right)"
            )

            st.markdown(
                """
            Où:
            - **N** = nombre total de documents (ici: {})
            - **n(t)** = nombre de docs contenant le terme t
            - **+0.5** = smoothing de Laplace
            """.format(mini_bm25.N)
            )

            # Calculer IDF pour chaque mot de la query
            idf_data = []
            for word in query_words:
                if word in mini_bm25.word_to_idx:
                    idx = mini_bm25.word_to_idx[word]
                    n_t = mini_bm25.doc_count[idx]
                    idf = np.log((mini_bm25.N - n_t + 0.5) / (n_t + 0.5))

                    idf_data.append(
                        {
                            "Mot": word,
                            "n(t)": n_t,
                            "Calcul": f"log(({mini_bm25.N} - {n_t} + 0.5) / ({n_t} + 0.5))",
                            "IDF": f"{idf:.4f}",
                        }
                    )
                else:
                    idf_data.append(
                        {"Mot": word, "n(t)": 0, "Calcul": "Mot absent", "IDF": "N/A"}
                    )

            df_idf = pd.DataFrame(idf_data)
            st.dataframe(df_idf, use_container_width=True)

            st.markdown("""
            ### 💡 Interprétation

            - **IDF élevé** → Mot rare → Plus important!
            - **IDF faible** → Mot commun → Moins important
            - Le smoothing (+0.5) évite les divisions par zéro et stabilise les valeurs
            """)

        # === ÉTAPE 2: FRÉQUENCES ===
        with st.expander("**🔢 Étape 2: Comptage des Fréquences**", expanded=False):
            st.markdown("""
            ### Comptons combien de fois chaque mot de la query apparaît dans chaque document!
            """)

            freq_data = []
            for doc_idx, text in enumerate(sample_texts):
                row = {"Document": sample_titles[doc_idx][:30]}
                doc_words = preprocess_text(text, remove_stopwords=remove_stopwords)

                for word in query_words:
                    count = doc_words.count(word)
                    row[word] = count

                freq_data.append(row)

            df_freq = pd.DataFrame(freq_data)
            st.dataframe(df_freq, use_container_width=True)

            st.info("""
            📌 **Note:** Ce sont les fréquences brutes (nombre d'occurrences).
            BM25 va les **saturer** avec le paramètre k1!
            """)

        # === ÉTAPE 3: NORMALISATION ===
        with st.expander("**⚖️ Étape 3: Facteurs de Normalisation**", expanded=False):
            st.markdown("""
            ### Formule de Normalisation
            """)

            st.latex(r"\text{norm}(D) = 1 - b + b \times \frac{|D|}{\text{avgdl}}")

            st.markdown(f"""
            Paramètres:
            - **b** = {b_tutorial} (intensité de la pénalité)
            - **avgdl** = {mini_bm25.avgdl:.1f} mots
            """)

            norm_data = []
            for doc_idx, text in enumerate(sample_texts):
                doc_length = len(text.split())
                norm_factor = (
                    1 - b_tutorial + b_tutorial * (doc_length / mini_bm25.avgdl)
                )

                norm_data.append(
                    {
                        "Document": sample_titles[doc_idx][:30],
                        "|D| (longueur)": doc_length,
                        "Calcul": f"1 - {b_tutorial} + {b_tutorial} × ({doc_length}/{mini_bm25.avgdl:.1f})",
                        "Facteur norm": f"{norm_factor:.3f}",
                    }
                )

            df_norm = pd.DataFrame(norm_data)
            st.dataframe(df_norm, use_container_width=True)

            st.markdown("""
            ### 💡 Interprétation

            - **norm > 1** → Document **pénalisé** (plus long que la moyenne)
            - **norm = 1** → Document de longueur moyenne
            - **norm < 1** → Document **boosté** (plus court que la moyenne)

            Ce facteur sera utilisé dans le dénominateur de BM25!
            """)

        # === ÉTAPE 4: TF SATURÉ ===
        with st.expander("**🎛️ Étape 4: TF Saturé (avec k1)**", expanded=False):
            st.markdown("""
            ### Formule du TF Saturé BM25
            """)

            st.latex(
                r"\text{TF}_{\text{BM25}} = \frac{f \times (k1 + 1)}{f + k1 \times \text{norm}}"
            )

            st.markdown(f"""
            Où:
            - **f** = fréquence du mot dans le doc
            - **k1** = {k1_tutorial} (contrôle la saturation)
            - **norm** = facteur de normalisation (calculé à l'étape 3)
            """)

            # Calculer TF saturé pour chaque mot dans chaque doc
            tf_data = []
            for doc_idx, text in enumerate(sample_texts):
                doc_words = preprocess_text(text, remove_stopwords=remove_stopwords)
                doc_length = len(text.split())
                norm_factor = (
                    1 - b_tutorial + b_tutorial * (doc_length / mini_bm25.avgdl)
                )

                row = {"Document": sample_titles[doc_idx][:30]}

                for word in query_words:
                    f = doc_words.count(word)
                    tf_bm25 = (f * (k1_tutorial + 1)) / (f + k1_tutorial * norm_factor)
                    row[f"{word} (f={f})"] = f"{tf_bm25:.4f}"

                tf_data.append(row)

            df_tf = pd.DataFrame(tf_data)
            st.dataframe(df_tf, use_container_width=True)

            st.markdown("""
            ### 💡 Observation Clé

            Contrairement à TF-IDF (où TF = f/longueur), ici le TF **plafonne**!

            - Si f = 0 → TF = 0
            - Si f = 1 → TF ≈ 1.0
            - Si f = 10 → TF ≈ 1.3 (saturation!)
            - Si f = 100 → TF ≈ 1.5 (plateau atteint!)

            **C'est le cœur de BM25!** 🎯
            """)

        # === ÉTAPE 5: BM25 FINAL ===
        with st.expander("**🎯 Étape 5: Score BM25 Final (IDF × TF)**", expanded=False):
            st.markdown("""
            ### Formule Complète BM25
            """)

            st.latex(r"""
            \text{BM25}(D) = \sum_{t \in Q} \text{IDF}(t) \times \frac{f(t, D) \times (k1 + 1)}{f(t, D) + k1 \times \text{norm}(D)}
            """)

            st.markdown("""
            On **multiplie IDF × TF** pour chaque mot, puis on **additionne**!
            """)

            # Calculer BM25 complet
            bm25_data = []
            for doc_idx, text in enumerate(sample_texts):
                doc_words = preprocess_text(text, remove_stopwords=remove_stopwords)
                doc_length = len(text.split())
                norm_factor = (
                    1 - b_tutorial + b_tutorial * (doc_length / mini_bm25.avgdl)
                )

                total_score = 0
                details = []

                for word in query_words:
                    if word in mini_bm25.word_to_idx:
                        idx = mini_bm25.word_to_idx[word]

                        # IDF
                        n_t = mini_bm25.doc_count[idx]
                        idf = np.log((mini_bm25.N - n_t + 0.5) / (n_t + 0.5))

                        # TF saturé
                        f = doc_words.count(word)
                        tf = (f * (k1_tutorial + 1)) / (f + k1_tutorial * norm_factor)

                        # Score du mot
                        word_score = idf * tf
                        total_score += word_score

                        details.append(
                            f"{word}: {idf:.3f} × {tf:.3f} = {word_score:.4f}"
                        )

                bm25_data.append(
                    {
                        "Document": sample_titles[doc_idx][:30],
                        "Détail": " + ".join(details)
                        if details
                        else "Aucun mot trouvé",
                        "Score BM25": f"{total_score:.4f}",
                    }
                )

            df_bm25 = pd.DataFrame(bm25_data)
            st.dataframe(df_bm25, use_container_width=True, height=200)

            st.success("""
            ✅ **Voilà le score BM25 final!**
            Plus le score est élevé, plus le document est pertinent pour la query.
            """)

        # === ÉTAPE 6: CLASSEMENT ===
        with st.expander("**🏆 Étape 6: Classement Final**", expanded=True):
            st.markdown("""
            ### Classement par Score BM25 Décroissant
            """)

            results = mini_bm25.search(query, top_k=3)

            if len(results) == 0:
                st.warning("😕 Aucun résultat trouvé!")
            else:
                for rank, (doc_idx, score) in enumerate(results, 1):
                    if rank == 1:
                        medal = "🥇"
                    elif rank == 2:
                        medal = "🥈"
                    else:
                        medal = "🥉"

                    st.markdown(f"""
                    {medal} **#{rank} - {sample_titles[doc_idx]}**
                    Score BM25: **{score:.4f}**
                    """)

                    # Snippet
                    st.caption(sample_texts[doc_idx][:150] + "...")

                st.divider()

                st.markdown("""
                ### 🎓 Conclusion

                Nous avons calculé BM25 **de A à Z**:

                1. ✅ **IDF** avec smoothing (rareté des mots)
                2. ✅ **Fréquences** brutes (comptage)
                3. ✅ **Normalisation** par longueur (facteur b)
                4. ✅ **TF saturé** (effet de plateau k1)
                5. ✅ **Multiplication** IDF × TF pour chaque mot
                6. ✅ **Classement** des documents

                **BM25 > TF-IDF** car il évite la saturation linéaire et permet de tuner les paramètres! 🚀
                """)


def render_bm25_comparison(
    documents_texts, documents_titles, tfidf_engine, remove_stopwords
):
    """Comparaison TF-IDF vs BM25 - ENRICHIE"""
    st.header("⚔️ TF-IDF vs BM25: Le Duel!")

    st.markdown("""
    ### 🎯 Objectif

    Comparons les **deux algorithmes** sur la **même requête** pour voir:
    - Quels documents sont retrouvés par chacun
    - Comment les scores diffèrent
    - Quel algorithme performe mieux
    """)

    query_compare = st.text_input(
        "🔎 Requête de comparaison:",
        value="recette italienne pâtes fromage",
        key="compare_query",
        help="💡 Teste avec plusieurs mots pour voir la différence!",
    )

    top_k_compare = st.slider("Nombre de résultats:", 5, 20, 10, key="compare_topk")

    if query_compare and st.button(
        "⚔️ Lancer la Comparaison!", type="primary", key="compare_btn"
    ):
        with st.spinner("⚔️ Comparaison en cours..."):
            import time

            start_tfidf = time.time()
            tfidf_results = tfidf_engine.search(query_compare, top_k=top_k_compare)
            time_tfidf = (time.time() - start_tfidf) * 1000  # ms

            start_bm25 = time.time()
            bm25_engine = BM25Engine(
                documents_texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
            )
            bm25_results = bm25_engine.search(query_compare, top_k=top_k_compare)
            time_bm25 = (time.time() - start_bm25) * 1000  # ms

            # === MÉTRIQUES RAPIDES ===
            st.markdown("## 📊 Métriques Globales")

            col1, col2, col3, col4 = st.columns(4)

            tfidf_indices = set([idx for idx, _ in tfidf_results])
            bm25_indices = set([idx for idx, _ in bm25_results])
            overlap = len(tfidf_indices.intersection(bm25_indices))

            col1.metric("⏱️ TF-IDF", f"{time_tfidf:.2f} ms")
            col2.metric("⏱️ BM25", f"{time_bm25:.2f} ms")
            col3.metric(
                "📊 Overlap",
                f"{overlap}/{top_k_compare}",
                f"{(overlap / top_k_compare * 100):.0f}%",
            )
            col4.metric(
                "🎯 Concordance",
                "Élevée"
                if overlap > top_k_compare * 0.7
                else "Moyenne"
                if overlap > top_k_compare * 0.4
                else "Faible",
            )

            st.divider()

            # === VISUALISATION COMPARATIVE ===
            st.markdown("## 📈 Comparaison Visuelle des Scores")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                fig_comp = plot_tfidf_bm25_comparison(
                    tfidf_results,
                    bm25_results,
                    documents_titles,
                    query_compare,
                    top_k=top_k_compare,
                )
                st.pyplot(fig_comp)

            with col_g2:
                st.markdown("### 📊 Analyse du Graphique")

                # Analyses statistiques
                tfidf_scores = [score for _, score in tfidf_results]
                bm25_scores = [score for _, score in bm25_results]

                tfidf_max = max(tfidf_scores)
                bm25_max = max(bm25_scores)
                tfidf_range = max(tfidf_scores) - min(tfidf_scores)
                bm25_range = max(bm25_scores) - min(bm25_scores)

                st.markdown(f"""
                **TF-IDF:**
                - Score max: **{tfidf_max:.4f}**
                - Écart: **{tfidf_range:.4f}**

                **BM25:**
                - Score max: **{bm25_max:.4f}**
                - Écart: **{bm25_range:.4f}**

                **Observations:**
                """)

                if bm25_range > tfidf_range * 1.2:
                    st.success(
                        "✅ **BM25 a une meilleure séparation** des scores! Les résultats sont plus différenciés."
                    )
                elif tfidf_range > bm25_range * 1.2:
                    st.info(
                        "💡 **TF-IDF a une meilleure séparation** pour cette query."
                    )
                else:
                    st.info("💡 **Séparation similaire** entre les deux algorithmes.")

            st.divider()

            # === ANALYSE DES DIFFÉRENCES ===
            st.markdown("## 🔍 Analyse des Différences")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔴 Uniques à TF-IDF")
                tfidf_unique = tfidf_indices - bm25_indices
                if len(tfidf_unique) > 0:
                    for idx in list(tfidf_unique)[:3]:
                        tfidf_score = next(
                            score for doc_idx, score in tfidf_results if doc_idx == idx
                        )
                        st.markdown(
                            f"- **{documents_titles[idx][:40]}** (score: {tfidf_score:.4f})"
                        )
                    st.caption(f"Total: {len(tfidf_unique)} documents uniques")
                else:
                    st.info("Aucun document unique!")

            with col2:
                st.markdown("### 🟢 Uniques à BM25")
                bm25_unique = bm25_indices - tfidf_indices
                if len(bm25_unique) > 0:
                    for idx in list(bm25_unique)[:3]:
                        bm25_score = next(
                            score for doc_idx, score in bm25_results if doc_idx == idx
                        )
                        st.markdown(
                            f"- **{documents_titles[idx][:40]}** (score: {bm25_score:.4f})"
                        )
                    st.caption(f"Total: {len(bm25_unique)} documents uniques")
                else:
                    st.info("Aucun document unique!")

            st.divider()

            # === DISTRIBUTION DES SCORES ===
            st.markdown("## 📊 Distribution des Scores")

            col_dist1, col_dist2 = st.columns(2)

            with col_dist1:
                fig_dist = plot_score_distributions(tfidf_scores, bm25_scores)
                st.pyplot(fig_dist)

            with col_dist2:
                st.markdown("### 📈 Interprétation")

                st.markdown("""
                **Histogrammes:**
                - Montrent la **répartition** des scores
                - TF-IDF (rouge) vs BM25 (vert)

                **Idéal:**
                - Distribution **étalée** (bonne séparation)
                - Pic à gauche (faibles scores)
                - Queue à droite (bons résultats)
                """)

                # Calcul de la variance pour mesurer la dispersion
                import numpy as np

                var_tfidf = np.var(tfidf_scores)
                var_bm25 = np.var(bm25_scores)

                if var_bm25 > var_tfidf * 1.2:
                    st.success(
                        "✅ **BM25 a une meilleure dispersion** → Résultats plus différenciés"
                    )
                elif var_tfidf > var_bm25 * 1.2:
                    st.info("💡 **TF-IDF a une meilleure dispersion** pour cette query")
                else:
                    st.info("💡 **Dispersion similaire**")

            st.divider()

            # === CONCLUSION ===
            st.markdown("## 🎓 Conclusion")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 🔴 TF-IDF

                **Avantages:**
                - ✅ Simple à comprendre
                - ✅ Rapide à calculer
                - ✅ Pas de paramètres

                **Inconvénients:**
                - ❌ Saturation linéaire
                - ❌ Normalisation rigide
                - ❌ Pas ajustable
                """)

            with col2:
                st.markdown("""
                ### 🟢 BM25

                **Avantages:**
                - ✅ Saturation intelligente (k1)
                - ✅ Normalisation ajustable (b)
                - ✅ IDF amélioré (smoothing)
                - ✅ Standard industriel

                **Inconvénients:**
                - ⚠️ Légèrement plus complexe
                - ⚠️ Nécessite tuning des paramètres
                """)

            overlap_pct = (overlap / top_k_compare) * 100
            if overlap_pct > 70:
                st.success(f"""
                ✅ **Accord élevé ({overlap_pct:.0f}%):** Les deux algorithmes sont cohérents!
                BM25 apporte surtout une **meilleure séparation** des scores.
                """)
            elif overlap_pct > 40:
                st.info(f"""
                💡 **Accord modéré ({overlap_pct:.0f}%):** Les algorithmes diffèrent!
                BM25 trouve des documents que TF-IDF manque (et vice-versa).
                """)
            else:
                st.warning(f"""
                ⚠️ **Accord faible ({overlap_pct:.0f}%):** Résultats très différents!
                Cela peut indiquer que **BM25 est plus adapté** à ce corpus.
                """)


def render_bm25_performance(documents_texts, remove_stopwords):
    """Performance BM25 - HYPER DÉTAILLÉE"""
    st.header("⚡ Performances BM25: Analyse Complète")

    st.markdown("""
    Analysons en profondeur les **performances de BM25** et comparons-les avec TF-IDF!
    """)

    # === COMPLEXITÉ ALGORITHMIQUE ===
    st.markdown("## 🧮 Complexité Algorithmique")

    st.markdown("""
    **Question importante:** BM25 est-il plus lent que TF-IDF? 🤔
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 🔴 TF-IDF

        **Preprocessing (indexation):**
        - Compter les mots: O(n × m)
        - Calculer IDF: O(v)
        - Calculer TF-IDF: O(n × v)

        **Recherche:**
        - Vectoriser query: O(|q|)
        - Cosine similarity: O(n × v)

        **Total:** O(n × m + n × v)
        """)

    with col2:
        st.success("""
        ### 🟢 BM25

        **Preprocessing (indexation):**
        - Compter les mots: O(n × m)
        - Calculer avgdl: O(n)
        - Calculer IDF: O(v)

        **Recherche:**
        - Calculer BM25: O(n × |q|)

        **Total:** O(n × m + n × |q|)

        ✅ **Identique!**
        """)

    st.markdown("""
    ### 💡 Pourquoi Même Complexité?

    Les calculs supplémentaires de BM25 sont en **O(1)** par terme:
    - **norm_factor** = `1 - b + b × (|D| / avgdl)` → **O(1)**
    - **TF saturé** = `f × (k1 + 1) / (f + k1 × norm)` → **O(1)**

    Ces multiplications/divisions sont **négligeables** par rapport au comptage des mots!

    **Conclusion:** BM25 n'est **PAS** plus lent que TF-IDF en pratique! 🚀
    """)

    st.divider()

    # === MÉTRIQUES DU CORPUS ACTUEL ===
    st.markdown("## 📊 Métriques du Corpus Actuel")

    import time

    # Mesurer le temps de chargement
    start_load = time.time()
    n_docs = len(documents_texts)
    total_words = sum(len(doc.split()) for doc in documents_texts)
    avg_length = total_words / n_docs if n_docs > 0 else 0
    time_load = (time.time() - start_load) * 1000

    # Mesurer l'indexation BM25
    start_index = time.time()
    bm25_engine = BM25Engine(
        documents_texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
    )
    time_index = (time.time() - start_index) * 1000

    vocab_size = len(bm25_engine.vocabulary)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📚 Documents", f"{n_docs:,}")
    col2.metric("📖 Vocabulaire", f"{vocab_size:,}")
    col3.metric("📏 Longueur moy.", f"{avg_length:.1f} mots")
    col4.metric("💬 Total mots", f"{total_words:,}")

    st.divider()

    col_t1, col_t2, col_t3 = st.columns(3)

    col_t1.metric("⏱️ Chargement", f"{time_load:.2f} ms")
    col_t2.metric("⏱️ Indexation BM25", f"{time_index:.2f} ms")

    # Efficacité
    docs_per_sec = (n_docs / time_index * 1000) if time_index > 0 else 0
    if docs_per_sec > 1000:
        efficiency = "Ultra rapide! 🚀"
        col_t3.metric("🚀 Efficacité", f"{docs_per_sec:.0f} docs/s", delta="Excellent")
    elif docs_per_sec > 100:
        efficiency = "Rapide! ✅"
        col_t3.metric("⚡ Efficacité", f"{docs_per_sec:.0f} docs/s", delta="Bon")
    else:
        efficiency = "Lent... ⚠️"
        col_t3.metric("🐌 Efficacité", f"{docs_per_sec:.0f} docs/s", delta="Améliorer")

    st.info(f"""
    💡 **Interprétation:** {efficiency}
    BM25 a indexé {n_docs:,} documents en **{time_index:.2f} ms** ({docs_per_sec:.0f} docs/s).
    """)

    st.divider()

    # === EXPLICATION COMPLEXITÉ ===
    st.markdown("## 📖 Comprendre la Complexité O(n × m)")

    st.markdown("""
    ### Que signifient **n** et **m**?

    - **n** = nombre de documents dans le corpus
    - **m** = longueur moyenne d'un document (en mots)
    - **v** = taille du vocabulaire (mots uniques)
    - **|q|** = longueur de la query
    """)

    st.markdown(f"""
    ### Pour ton corpus actuel:

    - **n** = {n_docs:,} documents
    - **m** = {avg_length:.0f} mots/doc (moyenne)
    - **v** = {vocab_size:,} mots uniques

    **Complexité d'indexation:** O({n_docs:,} × {avg_length:.0f}) = O({n_docs * avg_length:,.0f}) opérations
    """)

    with st.expander("🔍 Voir le Détail des Opérations"):
        st.markdown(f"""
        ### Étapes de l'Indexation BM25

        **1. Tokenisation (parcourir tous les mots):**
        - {n_docs:,} docs × {avg_length:.0f} mots = **{n_docs * avg_length:,.0f} mots** à traiter
        - Complexité: **O(n × m)**

        **2. Construction du vocabulaire:**
        - Ajouter chaque mot unique dans un dictionnaire
        - Complexité: **O(n × m)** (pire cas)
        - Résultat: **{vocab_size:,} mots uniques**

        **3. Calcul de avgdl (longueur moyenne):**
        - Somme des longueurs / nombre de docs
        - Complexité: **O(n)**
        - Résultat: **avgdl = {bm25_engine.avgdl:.1f} mots**

        **4. Comptage des documents contenant chaque mot (pour IDF):**
        - Pour chaque mot, compter dans combien de docs il apparaît
        - Complexité: **O(n × v)** (pire cas)

        **Total:** O(n × m + n × v) ≈ **O(n × m)** (car généralement m > v pour un doc)
        """)

    st.markdown(r"""
    ### 📈 Impact de Doubler n ou m

    | Action | Impact sur Temps | Exemple |
    |--------|------------------|---------|
    | **n × 2** (doubler les docs) | **Temps × 2** | 1000 → 2000 docs |
    | **m × 2** (docs 2× plus longs) | **Temps × 2** | 100 → 200 mots/doc |
    | **v × 2** (vocabulaire 2× plus grand) | Impact faible | 5000 → 10000 mots |
    | **\|q\| × 2** (query 2× plus longue) | Impact faible (recherche only) | 3 → 6 mots |

    **Conclusion:** Le nombre de documents **n** et leur longueur **m** sont les facteurs clés!
    """)

    st.divider()

    # === BENCHMARKS AUTOMATIQUES ===
    st.markdown("## 🏁 Benchmarks Automatiques Multi-Datasets")

    st.markdown("""
    Testons BM25 sur **différents datasets** pour voir l'impact de la taille du corpus!
    """)

    if st.button("🚀 Lancer les Benchmarks!", type="primary", key="bm25_bench_btn"):
        with st.spinner("⏳ Benchmarks en cours... (peut prendre quelques secondes)"):
            from src.data_loader import load_dataset

            benchmark_results = []

            # Définir les datasets à tester (petits échantillons)
            test_configs = [
                ("recettes", False, 50),
                ("films", False, 50),
                ("wikipedia", False, 200),
                ("recettes", True, None),  # Version étendue
            ]

            for dataset_name, extended, sample_size in test_configs:
                try:
                    # Charger le dataset
                    start = time.time()
                    dataset = load_dataset(
                        dataset_name, extended=extended, sample_size=sample_size
                    )
                    time_load_bench = (time.time() - start) * 1000

                    if len(dataset) == 0:
                        continue

                    texts = [doc["text"] for doc in dataset]
                    n_bench = len(texts)

                    # Indexation BM25
                    start = time.time()
                    bm25_bench = BM25Engine(
                        texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
                    )
                    time_index_bench = (time.time() - start) * 1000

                    # Recherche test
                    test_query = "italien fromage"
                    start = time.time()
                    _ = bm25_bench.search(test_query, top_k=10)
                    time_search = (time.time() - start) * 1000

                    vocab_bench = len(bm25_bench.vocabulary)

                    benchmark_results.append(
                        {
                            "Dataset": f"{dataset_name} {'(étendu)' if extended else ''}",
                            "Docs": n_bench,
                            "Vocabulaire": vocab_bench,
                            "Load (ms)": f"{time_load_bench:.2f}",
                            "Index (ms)": f"{time_index_bench:.2f}",
                            "Search (ms)": f"{time_search:.2f}",
                            "Total (ms)": f"{(time_load_bench + time_index_bench + time_search):.2f}",
                        }
                    )

                except Exception as e:
                    st.warning(f"⚠️ Erreur avec {dataset_name}: {str(e)}")
                    continue

            if len(benchmark_results) > 0:
                # Afficher les résultats
                st.markdown("### 📊 Résultats des Benchmarks")

                df_bench = pd.DataFrame(benchmark_results)
                st.dataframe(df_bench, use_container_width=True)

                # Analyse automatique
                st.markdown("### 📈 Analyse des Résultats")

                # Graphique: Temps d'indexation vs nombre de docs
                import matplotlib.pyplot as plt

                docs_list = [int(r["Docs"]) for r in benchmark_results]
                index_times = [float(r["Index (ms)"]) for r in benchmark_results]

                col_plot1, col_plot2 = st.columns(2)

                with col_plot1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(docs_list, index_times, s=100, alpha=0.7, color="green")
                    ax.plot(docs_list, index_times, "--", alpha=0.5, color="green")
                    ax.set_xlabel("Nombre de Documents", fontsize=11)
                    ax.set_ylabel("Temps d'Indexation (ms)", fontsize=11)
                    ax.set_title(
                        "BM25: Temps vs Nombre de Docs", fontsize=12, fontweight="bold"
                    )
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

                with col_plot2:
                    st.markdown("### 📊 Observations")

                    if len(docs_list) >= 2:
                        # Calculer une tendance linéaire simple
                        from numpy import polyfit

                        coeffs = polyfit(docs_list, index_times, 1)
                        slope = coeffs[0]

                        st.markdown(f"""
                        **Tendance:**
                        - Pente: **{slope:.4f} ms/doc**
                        - Relation: **Quasi-linéaire** ✅

                        **Interprétation:**
                        - Doubler le nombre de docs ≈ doubler le temps
                        - Confirme la complexité **O(n)**!

                        **Vitesse moyenne:**
                        - **{(len(docs_list) / sum(index_times) * 1000):.0f} docs/s**
                        """)

                    # Dataset le plus rapide/lent
                    fastest_idx = index_times.index(min(index_times))
                    slowest_idx = index_times.index(max(index_times))

                    st.success(f"""
                    ⚡ **Plus rapide:** {benchmark_results[fastest_idx]["Dataset"]}
                    ({benchmark_results[fastest_idx]["Docs"]} docs en {benchmark_results[fastest_idx]["Index (ms)"]} ms)
                    """)

                    st.warning(f"""
                    🐌 **Plus lent:** {benchmark_results[slowest_idx]["Dataset"]}
                    ({benchmark_results[slowest_idx]["Docs"]} docs en {benchmark_results[slowest_idx]["Index (ms)"]} ms)
                    """)

                st.divider()

                st.markdown("### 🎓 Conclusion des Benchmarks")

                max_docs = max(docs_list)
                min_docs = min(docs_list)
                max_time = max(index_times)
                min_time = min(index_times)

                ratio_docs = max_docs / min_docs
                ratio_time = max_time / min_time

                st.markdown(f"""
                **Scalabilité de BM25:**

                - En passant de **{min_docs} à {max_docs} docs** (×{ratio_docs:.1f}),
                  le temps passe de **{min_time:.2f} à {max_time:.2f} ms** (×{ratio_time:.1f})

                **Observations:**
                """)

                if abs(ratio_time - ratio_docs) < 0.5:
                    st.success(f"""
                    ✅ **Scalabilité linéaire confirmée!**
                    Le ratio de temps ({ratio_time:.1f}×) correspond au ratio de docs ({ratio_docs:.1f}×).
                    BM25 respecte bien la complexité O(n)!
                    """)
                else:
                    st.info(f"""
                    💡 **Scalabilité observée:** Ratio temps ({ratio_time:.1f}×) vs ratio docs ({ratio_docs:.1f}×).
                    Les variations peuvent être dues aux longueurs de documents différentes.
                    """)

            else:
                st.error("❌ Aucun benchmark n'a pu être exécuté!")

    st.divider()

    # === OPTIMISATIONS ===
    st.markdown("## 🔧 Optimisations Possibles")

    st.markdown("""
    Pour de **très gros corpus** (millions de documents), voici des optimisations possibles:
    """)

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.markdown("""
        ### 1️⃣ Index Inversé

        **Principe:** Stocker pour chaque mot la liste des documents qui le contiennent.

        **Avantage:** Ne parcourir que les docs pertinents (pas tout le corpus)!

        **Gain:** O(n) → O(k) où k = docs contenant les mots de la query

        ---

        ### 2️⃣ Matrices Creuses (Sparse)

        **Principe:** Ne stocker que les valeurs non-nulles.

        **Avantage:** Économie mémoire massive!

        **Gain:** Mémoire O(n × v) → O(nnz) où nnz = valeurs non-nulles

        ---

        ### 3️⃣ Cache de Preprocessing

        **Principe:** Sauvegarder l'index BM25 sur disque.

        **Avantage:** Pas besoin de réindexer à chaque run!

        **Gain:** Temps de démarrage divisé par 10-100×
        """)

    with col_opt2:
        st.markdown("""
        ### 4️⃣ Batch Processing

        **Principe:** Traiter les documents par lots parallèles.

        **Avantage:** Utiliser tous les CPU cores!

        **Gain:** Temps divisé par le nombre de cores (4-16×)

        ---

        ### 5️⃣ Approximations (ANN)

        **Principe:** Approximate Nearest Neighbors (LSH, HNSW).

        **Avantage:** Recherche ultra-rapide (log n au lieu de n)!

        **Gain:** O(n) → O(log n) pour la recherche

        ---

        ### 6️⃣ Elasticsearch / Solr

        **Principe:** Utiliser un moteur dédié (basé sur BM25!)

        **Avantage:** Toutes les optimisations ci-dessus + distribution!

        **Gain:** Scalabilité jusqu'à des milliards de docs
        """)

    st.success("""
    ✅ **Pour ce projet pédagogique:**
    L'implémentation actuelle est suffisante pour des corpus de **quelques milliers de documents**.
    Pour de la production, utilise **Elasticsearch** ou **Apache Solr** (qui implémentent BM25 nativement)!

    ---

    Compare les **performances réelles** de TF-IDF vs BM25 sur 100 documents!
    """)

    if st.button("🚀 Lancer le Benchmark!", type="primary", key="bm25_benchmark_btn"):
        with st.spinner("⏱️ Benchmarking en cours..."):
            # TF-IDF
            start = time.time()
            tfidf_engine = TFIDFEngine(
                documents_texts[:100], remove_stopwords=remove_stopwords
            )
            tfidf_engine.fit()
            tfidf_time = time.time() - start

            # BM25
            start = time.time()
            bm25_engine = BM25Engine(
                documents_texts[:100], remove_stopwords=remove_stopwords
            )
            bm25_time = time.time() - start

            st.success("✅ Benchmark terminé!")

            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ TF-IDF", f"{tfidf_time:.4f}s")
            col2.metric("⏱️ BM25", f"{bm25_time:.4f}s")

            # Différence avec indicateur
            diff = abs(bm25_time - tfidf_time)
            diff_percent = (diff / tfidf_time) * 100 if tfidf_time > 0 else 0
            col3.metric("📊 Différence", f"{diff:.4f}s", f"{diff_percent:.1f}%")

            st.info(
                "💡 **Conclusion:** Les deux algos ont la même complexité! BM25 apporte de meilleurs résultats sans pénalité de performance!"
            )


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
            datasets_info = get_all_datasets_info()
            dataset_names = [info["name"] for info in datasets_info]
            dataset_labels = {
                "recettes": "🍝 Recettes",
                "films": "🎬 Films",
                "wikipedia": "📚 Wikipedia",
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
                # Compter rapidement le nombre de docs
                if selected_dataset in ["recettes", "films"]:
                    # Lire depuis synthetic/
                    file_mapping = {
                        "recettes": "data/synthetic/recipes_fr.json",
                        "films": "data/synthetic/films_fr.json",
                    }
                    import json
                    from pathlib import Path

                    if use_extended:
                        # Mode étendu = TOUS les docs du fichier
                        file_path = Path(file_mapping[selected_dataset])
                        if file_path.exists():
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                estimated_docs = f"{len(data):,}"
                                size_label = "(étendu)"
                        else:
                            estimated_docs = "~1,000"
                            size_label = "(étendu)"
                    else:
                        # Mode normal = 50 docs
                        estimated_docs = "50"
                        size_label = ""

                elif selected_dataset == "wikipedia":
                    if use_extended:
                        estimated_docs = "1,000"
                        size_label = "(étendu - HF)"
                    else:
                        estimated_docs = "50"
                        size_label = "(hardcodé)"
                else:
                    estimated_docs = "?"
                    size_label = ""

                st.info(f"📊 **{estimated_docs} documents** {size_label}")

            except Exception as e:
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
