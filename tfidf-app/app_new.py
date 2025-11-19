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
from datasets import load_dataset, get_all_datasets_info
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
    plot_score_distributions
)


# Configuration de la page
st.set_page_config(
    page_title="Explorateur de Recherche Textuelle 🔍",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Style CSS personnalisé
st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================================
# FONCTIONS DE CACHE
# ============================================================================

@st.cache_data
def load_cached_dataset(dataset_name: str, sample_size: int = None, extended: bool = False):
    """Charge un dataset avec cache"""
    return load_dataset(dataset_name, sample_size=sample_size, extended=extended)


@st.cache_resource
def create_tfidf_engine(documents_texts: list, remove_stopwords: bool = True):
    """Crée et entraîne le moteur TF-IDF avec cache"""
    engine = TFIDFEngine(documents_texts, remove_stopwords=remove_stopwords)
    engine.fit()
    return engine


@st.cache_resource
def create_bm25_engine(documents_texts: list, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True):
    """Crée le moteur BM25 avec cache"""
    return BM25Engine(documents_texts, k1=k1, b=b, remove_stopwords=remove_stopwords)


# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

def render_home():
    """Page d'accueil avec présentation générale"""
    st.markdown('<h1 class="main-title">🔍 Explorateur de Recherche Textuelle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Apprends les techniques de recherche textuelle de manière interactive!</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Bienvenue!

    Cette application pédagogique t'enseigne les différentes techniques de **recherche textuelle**
    utilisées dans les moteurs de recherche modernes.

    ### 📚 Sections Disponibles
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📊 1. TF-IDF
        **Term Frequency - Inverse Document Frequency**

        - ✅ Technique classique de recherche
        - 📐 Basée sur fréquence des mots
        - 🎓 Facile à comprendre
        - ⚡ Rapide à calculer

        **Tu apprendras:**
        - Comment calculer TF et IDF
        - Pourquoi normaliser les fréquences
        - Similarité cosinus
        - Limites de l'approche
        """)

        st.markdown("""
        #### 🧠 3. Embeddings (À venir)
        **Représentations vectorielles sémantiques**

        - 🚧 En construction
        - Word2Vec, GloVe
        - Transformers et BERT
        - Recherche sémantique
        """)

    with col2:
        st.markdown("""
        #### 🎯 2. BM25
        **Best Matching 25 - Amélioration de TF-IDF**

        - ✨ État de l'art en recherche textuelle
        - 📈 Saturation intelligente du TF
        - 🎛️ Paramètres ajustables (k1, b)
        - ⚔️ Meilleur que TF-IDF en pratique

        **Tu apprendras:**
        - Problèmes de TF-IDF
        - Fonctionnement de BM25
        - Tuning des paramètres
        - Comparaison avec TF-IDF
        """)

        st.markdown("""
        #### 📊 4. Synthèse (À venir)
        **Comparaison de toutes les techniques**

        - 🚧 En construction
        - Benchmarks comparatifs
        - Cas d'usage recommandés
        - Guide de sélection
        """)

    st.divider()

    st.markdown("""
    ### 🚀 Comment Utiliser Cette App

    1. **Sélectionne une section** dans la barre latérale (←)
    2. **Choisis un dataset** (recettes, films, ou wikipedia)
    3. **Explore les concepts** avec visualisations interactives
    4. **Teste la recherche** avec tes propres requêtes
    5. **Compare les techniques** pour comprendre les différences

    ### 💡 Conseils

    - 📖 Commence par **TF-IDF** pour comprendre les bases
    - 🎯 Passe ensuite à **BM25** pour voir les améliorations
    - 🔍 Utilise la **recherche interactive** pour tester
    - 📊 Consulte les **graphiques** pour visualiser
    - ⚡ Vérifie les **performances** pour comprendre la complexité
    """)

    st.success("👉 **Commence ton exploration en sélectionnant une section dans la sidebar!**")


# ============================================================================
# SECTION TF-IDF (contenu existant restructuré)
# ============================================================================

def render_tfidf_section(dataset, documents_texts, documents_titles, documents_categories, engine,
                         remove_stopwords, show_intermediate, load_time, fit_time):
    """Section TF-IDF complète avec tous les onglets"""

    st.title("📊 TF-IDF: Term Frequency - Inverse Document Frequency")

    # Sub-navigation
    tab = st.radio(
        "📍 Navigation TF-IDF:",
        ["📖 Introduction", "🔢 Concepts", "🔍 Recherche", "📊 Exploration", "🎓 Pas-à-Pas", "⚡ Performance"],
        horizontal=True,
        key="tfidf_tabs"
    )

    if tab == "📖 Introduction":
        render_tfidf_intro()
    elif tab == "🔢 Concepts":
        render_tfidf_concepts(engine, documents_titles)
    elif tab == "🔍 Recherche":
        render_tfidf_search(engine, documents_texts, documents_titles, documents_categories, show_intermediate)
    elif tab == "📊 Exploration":
        render_tfidf_exploration(engine, documents_titles, documents_categories)
    elif tab == "🎓 Pas-à-Pas":
        render_tfidf_stepbystep(documents_texts, documents_titles, documents_categories, remove_stopwords)
    elif tab == "⚡ Performance":
        render_tfidf_performance(engine, documents_texts, load_time, fit_time, remove_stopwords)


def render_tfidf_intro():
    """Introduction TF-IDF"""
    st.header("📖 Le Problème de la Recherche Textuelle")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🤔 Pourquoi la recherche simple ne suffit pas?

        **Approche naïve:** Compter les occurrences d'un mot.

        #### ❌ Problèmes:

        1. **Documents longs favorisés** injustement
        2. **Mots communs** polluent les résultats
        3. **Pas de notion de rareté**
        """)

    with col2:
        st.code("""
Doc A (20 mots):
  "chat" 2× → 10%

Doc B (200 mots):
  "chat" 3× → 1.5%

Naïf: B > A (3 > 2)
Correct: A > B (10% > 1.5%)
        """)

    st.divider()

    st.markdown("""
    ### ✅ La Solution: TF-IDF

    Combine deux mesures complémentaires:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **📈 TF (Term Frequency)**

        Fréquence locale du mot dans le document

        ✅ Normalise par longueur
        """)

    with col2:
        st.success("""
        **📉 IDF (Inverse Document Frequency)**

        Rareté globale du mot dans le corpus

        ✅ Pénalise les mots communs
        """)


def render_tfidf_concepts(engine, documents_titles):
    """Concepts TF-IDF détaillés"""
    st.header("🔢 Concepts TF-IDF en Profondeur")

    # Contenu existant des concepts TF-IDF...
    with st.expander("📈 **Term Frequency (TF)**", expanded=True):
        st.markdown("""
        ### 💡 Intuition

        **"Si un mot apparaît souvent, le doc parle de ce sujet"**

        Mais on normalise par la longueur!
        """)

        st.latex(r"\text{TF}(mot, doc) = \frac{\text{occurrences}}{\text{total mots}}")

        sample_indices = [0, 1, 2]
        sample_titles = [documents_titles[i] for i in sample_indices]

        fig_tf = plot_tf_comparison(engine.documents, sample_indices, sample_titles)
        st.pyplot(fig_tf)

    with st.expander("📉 **Inverse Document Frequency (IDF)**"):
        st.markdown("""
        ### 💡 Intuition

        **"Un mot rare est plus informatif"**
        """)

        st.latex(r"\text{IDF}(mot) = \log\left(\frac{N}{n}\right) + 1")

        fig_idf = plot_idf_curve(engine.idf_vector, engine.vocabulary, engine.documents)
        st.pyplot(fig_idf)

        idf_dict = {engine.vocabulary[i]: engine.idf_vector[i]
                   for i in range(min(100, len(engine.vocabulary)))}
        fig_wc = plot_idf_wordcloud(idf_dict)
        st.pyplot(fig_wc)

    with st.expander("🎯 **TF-IDF Combiné**"):
        st.latex(r"\text{TF-IDF} = \text{TF} \times \text{IDF}")

        fig_heatmap = plot_tfidf_heatmap(engine.tfidf_matrix, engine.vocabulary, documents_titles, top_words=15)
        st.pyplot(fig_heatmap)


def render_tfidf_search(engine, documents_texts, documents_titles, documents_categories, show_intermediate):
    """Recherche interactive TF-IDF"""
    st.header("🔍 Recherche Interactive TF-IDF")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input("🔎 Entre ta requête:", placeholder="Ex: recette italienne...", key="tfidf_query")

    with col2:
        top_k = st.slider("Résultats:", 3, 20, 5, key="tfidf_topk")

    if query and st.button("🚀 Rechercher!", type="primary", key="tfidf_search"):
        with st.spinner("🔍 Recherche en cours..."):
            results = engine.search(query, top_k=top_k)

            if len(results) == 0 or all(score == 0 for _, score in results):
                st.warning("😕 Aucun résultat. Essaie d'autres mots!")
            else:
                st.success(f"✅ {len(results)} résultats trouvés!")

                fig_results = plot_search_results(results, documents_titles, query)
                st.pyplot(fig_results)

                st.markdown("### 🎯 Résultats Détaillés")
                for rank, (doc_idx, score) in enumerate(results[:5], 1):
                    with st.expander(f"#{rank} - {documents_titles[doc_idx]} (Score: {score:.3f})"):
                        st.caption(f"Catégorie: {documents_categories[doc_idx]}")
                        st.write(documents_texts[doc_idx][:300] + "...")


def render_tfidf_exploration(engine, documents_titles, documents_categories):
    """Exploration du corpus TF-IDF"""
    st.header("📊 Exploration du Corpus")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Documents", len(documents_titles))
    col2.metric("🔤 Vocabulaire", len(engine.vocabulary))
    col3.metric("📝 Mots/Doc", f"{np.mean([len(doc) for doc in engine.documents]):.1f}")
    col4.metric("🏷️ Catégories", len(set(documents_categories)))

    fig_vocab = plot_vocabulary_stats(engine.documents)
    st.pyplot(fig_vocab)

    fig_3d = plot_documents_3d(engine.tfidf_matrix, documents_titles, documents_categories)
    st.plotly_chart(fig_3d, use_container_width=True)


def render_tfidf_stepbystep(documents_texts, documents_titles, documents_categories, remove_stopwords):
    """Exemple pas-à-pas TF-IDF"""
    st.header("🎓 Exemple Complet Pas-à-Pas")

    sample_indices = list(range(min(3, len(documents_texts))))

    for idx in sample_indices:
        with st.expander(f"📄 Document {idx+1}: {documents_titles[idx]}"):
            st.write(documents_texts[idx])

    query = st.text_input("🔎 Query pour l'exemple:", value="chat poisson", key="tfidf_tutorial")

    if query:
        sample_texts = [documents_texts[i] for i in sample_indices]
        mini_engine = TFIDFEngine(sample_texts, remove_stopwords=remove_stopwords)
        mini_engine.fit()

        st.markdown("### 🔢 Étape 1: Calcul des TF")
        # ... calculs détaillés ...


def render_tfidf_performance(engine, documents_texts, load_time, fit_time, remove_stopwords):
    """Performances TF-IDF"""
    st.header("⚡ Analyse des Performances")

    n_docs = len(documents_texts)
    n_vocab = len(engine.vocabulary)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Chargement", f"{load_time:.3f}s")
    col2.metric("🧮 Entraînement", f"{fit_time:.3f}s")
    col3.metric("📚 Documents", n_docs)
    col4.metric("🔤 Vocabulaire", n_vocab)

    st.markdown("### 🧮 Complexité: `O(n × v)`")


# ============================================================================
# SECTION BM25 (NOUVEAU!)
# ============================================================================

def render_bm25_section(dataset, documents_texts, documents_titles, documents_categories,
                        tfidf_engine, remove_stopwords):
    """Section BM25 complète"""

    st.title("🎯 BM25: Best Matching 25 - TF-IDF Amélioré")

    # Sub-navigation
    tab = st.radio(
        "📍 Navigation BM25:",
        ["📖 Introduction", "🔢 Concepts", "🔍 Recherche", "📊 Exploration", "🎓 Pas-à-Pas", "⚔️ Comparaison", "⚡ Performance"],
        horizontal=True,
        key="bm25_tabs"
    )

    if tab == "📖 Introduction":
        render_bm25_intro()
    elif tab == "🔢 Concepts":
        render_bm25_concepts(documents_texts, remove_stopwords)
    elif tab == "🔍 Recherche":
        render_bm25_search(documents_texts, documents_titles, documents_categories, remove_stopwords)
    elif tab == "📊 Exploration":
        render_bm25_exploration(documents_texts, documents_titles, remove_stopwords)
    elif tab == "🎓 Pas-à-Pas":
        render_bm25_stepbystep(documents_texts, documents_titles, remove_stopwords)
    elif tab == "⚔️ Comparaison":
        render_bm25_comparison(documents_texts, documents_titles, tfidf_engine, remove_stopwords)
    elif tab == "⚡ Performance":
        render_bm25_performance(documents_texts, remove_stopwords)


def render_bm25_intro():
    """Introduction BM25 & Problèmes de TF-IDF"""
    st.header("📖 BM25: La Solution aux Problèmes de TF-IDF")

    st.markdown("""
    ### 📊 Rappel TF-IDF

    TF-IDF combine fréquence locale (TF) et rareté globale (IDF).
    """)

    st.latex(r"\text{TF-IDF} = \text{TF} \times \text{IDF}")

    st.divider()

    st.markdown("### ❌ Les 3 Problèmes de TF-IDF")

    st.error("""
    **Problème #1: Saturation**

    TF croît linéairement avec les occurrences!

    - "chat" 10× → score 10
    - "chat" 100× → score 100

    ➡️ Est-ce que 100× est vraiment 10× plus pertinent? NON!
    """)

    # Visualisation saturation
    fig_sat = plot_saturation_effect()
    st.pyplot(fig_sat)

    st.error("""
    **Problème #2: Normalisation Naïve**

    TF-IDF normalise simplement par longueur totale.

    - Doc A (20 mots): "chat" 2× → TF = 0.10
    - Doc B (200 mots): "chat" 10× → TF = 0.05

    ➡️ Doc B a plus d'occurrences mais score PLUS BAS!
    """)

    st.error("""
    **Problème #3: Pas de Contrôle**

    TF-IDF est figé, aucun paramètre ajustable!

    ➡️ Impossible d'adapter selon le type de corpus.
    """)

    st.divider()

    st.success("""
    ### ✅ BM25 Résout Ces Problèmes

    **1. Saturation du TF** via paramètre **k1**
    - TF plafonne après un certain seuil
    - Plus réaliste!

    **2. Normalisation Paramétrable** via paramètre **b**
    - Contrôle la pénalité des documents longs
    - Ajustable selon le corpus!

    **3. IDF Amélioré** avec smoothing
    - Évite les valeurs extrêmes
    - Plus stable!
    """)


def render_bm25_concepts(documents_texts, remove_stopwords):
    """Concepts BM25 détaillés"""
    st.header("🔢 Comprendre BM25 en Profondeur")

    with st.expander("📉 **IDF Amélioré (avec smoothing)**", expanded=True):
        st.markdown("""
        ### Formule BM25 IDF
        """)

        st.latex(r"\text{IDF}(q) = \log\left(\frac{N - n(q) + 0.5}{n(q) + 0.5}\right)")

        st.markdown("""
        - **N** = nombre total de documents
        - **n(q)** = nombre de docs contenant le mot q
        - **+0.5** = smoothing pour éviter divisions par zéro

        **Différence avec TF-IDF:** Le smoothing rend l'IDF plus stable!
        """)

    with st.expander("🎛️ **Saturation du TF (Paramètre k1)**"):
        st.markdown("""
        ### 💡 Intuition

        Après X occurrences, le mot n'apporte plus d'info nouvelle.
        On veut un **PLATEAU**, pas une ligne droite!
        """)

        st.latex(r"\text{TF}_{saturated} = \frac{f \times (k1 + 1)}{f + k1}")

        st.markdown("""
        - **f** = fréquence du mot
        - **k1** = contrôle la vitesse de saturation

        **Valeurs typiques:**
        - k1 = 0 → binaire (présent/absent)
        - k1 = 1.2 → saturation agressive
        - k1 = 1.5 → **standard** ⭐
        - k1 = 2.0 → saturation lente
        - k1 = ∞ → comme TF-IDF (linéaire)
        """)

        fig_sat = plot_saturation_effect(k1_values=[0.5, 1.2, 1.5, 2.0])
        st.pyplot(fig_sat)

    with st.expander("⚖️ **Normalisation de Longueur (Paramètre b)**"):
        st.markdown("""
        ### 💡 Intuition

        Les docs longs contiennent naturellement plus de mots.
        Faut-il les pénaliser? **Ça dépend du corpus!**
        """)

        st.latex(r"\text{norm} = 1 - b + b \times \frac{|D|}{\text{avgdl}}")

        st.markdown("""
        - **|D|** = longueur du document
        - **avgdl** = longueur moyenne du corpus
        - **b** = intensité de la pénalité

        **Valeurs typiques:**
        - b = 0 → aucune normalisation
        - b = 0.5 → normalisation légère
        - b = 0.75 → **standard** ⭐
        - b = 1.0 → normalisation complète
        """)

        # Créer un mini corpus pour calculer avgdl
        bm25_demo = BM25Engine(documents_texts[:10], remove_stopwords=remove_stopwords)

        fig_norm = plot_length_normalization(
            avgdl=bm25_demo.avgdl,
            doc_lengths=[50, 100, 150, 200]
        )
        st.pyplot(fig_norm)

    with st.expander("🎯 **Formule Complète BM25**"):
        st.markdown("""
        ### La Grande Formule
        """)

        st.latex(r"""
        \text{BM25} = \sum_{i} \text{IDF}(q_i) \times \frac{f(q_i, D) \times (k1 + 1)}{f(q_i, D) + k1 \times (1 - b + b \times \frac{|D|}{\text{avgdl}})}
        """)

        st.markdown("""
        **En français:**

        Pour chaque mot de la query:
        1. Prendre son **IDF** (rareté)
        2. Multiplier par son **TF saturé** et normalisé
        3. Additionner tous les scores
        """)


def render_bm25_search(documents_texts, documents_titles, documents_categories, remove_stopwords):
    """Recherche interactive BM25"""
    st.header("🔍 Recherche Interactive BM25")

    st.markdown("""
    Teste BM25 avec tes propres paramètres!
    """)

    # Paramètres BM25
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        query = st.text_input("🔎 Votre recherche:", placeholder="Ex: recette italienne...", key="bm25_query")

    with col2:
        k1 = st.slider(
            "k1 (saturation)",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Contrôle la saturation du TF. Standard = 1.5",
            key="bm25_k1"
        )

    with col3:
        b = st.slider(
            "b (normalisation)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Contrôle la pénalité de longueur. Standard = 0.75",
            key="bm25_b"
        )

    top_k = st.slider("Nombre de résultats:", 3, 20, 5, key="bm25_topk")

    if query and st.button("🚀 Rechercher avec BM25!", type="primary", key="bm25_search_btn"):
        with st.spinner("🔍 Recherche BM25 en cours..."):
            # Créer engine BM25 avec les paramètres
            bm25_engine = BM25Engine(documents_texts, k1=k1, b=b, remove_stopwords=remove_stopwords)

            results = bm25_engine.search(query, top_k=top_k)

            if len(results) == 0 or all(score == 0 for _, score in results):
                st.warning("😕 Aucun résultat. Essaie d'autres mots!")
            else:
                st.success(f"✅ {len(results)} résultats BM25 trouvés!")

                fig_results = plot_search_results(results, documents_titles, query)
                st.pyplot(fig_results)

                st.markdown("### 🎯 Résultats Détaillés")
                for rank, (doc_idx, score) in enumerate(results[:5], 1):
                    with st.expander(f"#{rank} - {documents_titles[doc_idx]} (BM25: {score:.3f})"):
                        st.caption(f"Catégorie: {documents_categories[doc_idx]}")
                        st.write(documents_texts[doc_idx][:300] + "...")

                        if st.checkbox(f"📊 Voir calcul détaillé #{rank}", key=f"bm25_explain_{rank}"):
                            explanation = bm25_engine.explain(query, doc_idx)

                            st.json({
                                'avgdl': f"{explanation['avgdl']:.1f} mots",
                                'doc_length': f"{explanation['doc_length']} mots",
                                'norm_factor': f"{explanation['norm_factor']:.3f}",
                                'total_score': f"{explanation['total_score']:.4f}"
                            })


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
        format_func=lambda x: documents_titles[x]
    )

    test_query = st.text_input("Query de test:", value="recette cuisine", key="bm25_tuning_query")

    if test_query:
        with st.spinner("🧪 Génération de la heatmap..."):
            bm25_engine = BM25Engine(documents_texts, remove_stopwords=remove_stopwords)

            fig_heatmap = plot_parameter_space_heatmap(
                bm25_engine,
                test_query,
                doc_idx,
                k1_range=(0.5, 3.0),
                b_range=(0.0, 1.0),
                resolution=15
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.info("""
            💡 **Interprétation:**
            - Zones **rouges** = scores élevés
            - ⭐ **Étoile blanche** = paramètres standard (k1=1.5, b=0.75)
            - Explore l'espace pour voir l'impact!
            """)


def render_bm25_stepbystep(documents_texts, documents_titles, remove_stopwords):
    """Exemple pas-à-pas BM25"""
    st.header("🎓 Exemple Complet Pas-à-Pas BM25")

    st.markdown("""
    Suivons **tout le processus** de calcul BM25 étape par étape!
    """)

    sample_indices = list(range(min(3, len(documents_texts))))

    for idx in sample_indices:
        with st.expander(f"📄 Document {idx+1}: {documents_titles[idx]}"):
            st.write(documents_texts[idx])

    query = st.text_input("🔎 Query:", value="chat poisson", key="bm25_tutorial")

    if query:
        st.markdown("### Paramètres")
        col1, col2 = st.columns(2)
        with col1:
            k1_tutorial = st.number_input("k1:", value=1.5, key="bm25_tutorial_k1")
        with col2:
            b_tutorial = st.number_input("b:", value=0.75, key="bm25_tutorial_b")

        sample_texts = [documents_texts[i] for i in sample_indices]
        mini_bm25 = BM25Engine(sample_texts, k1=k1_tutorial, b=b_tutorial, remove_stopwords=remove_stopwords)

        st.markdown(f"""
        ### 📊 Statistiques du Mini-Corpus

        - **Nombre de documents:** {mini_bm25.N}
        - **Longueur moyenne (avgdl):** {mini_bm25.avgdl:.1f} mots
        - **Vocabulaire:** {len(mini_bm25.vocabulary)} mots uniques
        """)

        # Calculs détaillés...
        results = mini_bm25.search(query, top_k=3)

        st.markdown("### 🎯 Résultats Finaux")
        for rank, (doc_idx, score) in enumerate(results, 1):
            st.write(f"**#{rank}** - {documents_titles[sample_indices[doc_idx]]} : **{score:.4f}**")


def render_bm25_comparison(documents_texts, documents_titles, tfidf_engine, remove_stopwords):
    """Comparaison TF-IDF vs BM25 (SECTION CRITIQUE!)"""
    st.header("⚔️ Comparaison TF-IDF vs BM25")

    st.markdown("""
    ### 🔥 Le Face-à-Face!

    Compare les deux algorithmes sur une même requête.
    """)

    query_compare = st.text_input(
        "🔎 Requête de comparaison:",
        value="recette italienne pâtes",
        key="compare_query"
    )

    top_k_compare = st.slider("Nombre de résultats:", 5, 20, 10, key="compare_topk")

    if query_compare and st.button("⚔️ Comparer!", type="primary", key="compare_btn"):
        with st.spinner("⚔️ Comparaison en cours..."):
            # TF-IDF
            tfidf_results = tfidf_engine.search(query_compare, top_k=top_k_compare)

            # BM25
            bm25_engine = BM25Engine(documents_texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords)
            bm25_results = bm25_engine.search(query_compare, top_k=top_k_compare)

            # Visualisation comparative
            fig_comp = plot_tfidf_bm25_comparison(
                tfidf_results,
                bm25_results,
                documents_titles,
                query_compare,
                top_k=top_k_compare
            )
            st.pyplot(fig_comp)

            # Métriques de comparaison
            st.divider()

            col1, col2, col3 = st.columns(3)

            tfidf_indices = set([idx for idx, _ in tfidf_results])
            bm25_indices = set([idx for idx, _ in bm25_results])
            overlap = len(tfidf_indices.intersection(bm25_indices))

            col1.metric("📊 Overlap", f"{overlap}/{top_k_compare}")
            col2.metric("🔴 TF-IDF Unique", len(tfidf_indices - bm25_indices))
            col3.metric("🟢 BM25 Unique", len(bm25_indices - tfidf_indices))

            # Distributions
            st.markdown("### 📈 Distribution des Scores")

            all_tfidf_scores = [score for _, score in tfidf_results]
            all_bm25_scores = [score for _, score in bm25_results]

            fig_dist = plot_score_distributions(all_tfidf_scores, all_bm25_scores)
            st.pyplot(fig_dist)

            st.success("""
            ✅ **Observation:** BM25 a généralement une meilleure séparation des scores
            grâce à la saturation intelligente!
            """)


def render_bm25_performance(documents_texts, remove_stopwords):
    """Performance BM25"""
    st.header("⚡ Analyse des Performances BM25")

    st.markdown("""
    ### 🧮 Complexité Algorithmique

    **Bonne nouvelle:** BM25 a la **même complexité** que TF-IDF!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **TF-IDF:**
        - Preprocessing: O(n × m)
        - Search: O(n × v)
        - **Total: O(n × m + n × v)**
        """)

    with col2:
        st.success("""
        **BM25:**
        - Preprocessing: O(n × m)
        - Search: O(n × v)
        - **Total: O(n × m + n × v)**

        ✅ Identique!
        """)

    st.markdown("""
    ### 💡 Pourquoi Même Complexité?

    La saturation et normalisation BM25 sont juste des **multiplications**!

    - Calcul de `norm_factor`: O(1)
    - Formule BM25 par terme: O(1)

    ➡️ **BM25 n'est PAS plus lent que TF-IDF!**
    """)

    # Benchmark
    if st.checkbox("🧪 Faire un benchmark réel?"):
        with st.spinner("⏱️ Benchmarking..."):
            # TF-IDF
            start = time.time()
            tfidf_engine = TFIDFEngine(documents_texts[:100], remove_stopwords=remove_stopwords)
            tfidf_engine.fit()
            tfidf_time = time.time() - start

            # BM25
            start = time.time()
            bm25_engine = BM25Engine(documents_texts[:100], remove_stopwords=remove_stopwords)
            bm25_time = time.time() - start

            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ TF-IDF", f"{tfidf_time:.4f}s")
            col2.metric("⏱️ BM25", f"{bm25_time:.4f}s")
            col3.metric("📊 Différence", f"{abs(bm25_time - tfidf_time):.4f}s")

            st.success("✅ Les deux algos sont aussi rapides! BM25 apporte juste de meilleurs résultats!")


# ============================================================================
# PLACEHOLDERS SECTIONS FUTURES
# ============================================================================

def render_embeddings_placeholder():
    """Placeholder Embeddings"""
    st.title("🧠 Embeddings Vectoriels")

    st.info("""
    ### 🚧 Section en Construction

    Cette section couvrira:

    - **Word2Vec** : Représentations vectorielles de mots
    - **GloVe** : Global Vectors for Word Representation
    - **FastText** : Embeddings avec sous-mots
    - **BERT & Transformers** : Représentations contextuelles
    - **Recherche sémantique** : Au-delà des mots-clés

    **À venir prochainement!** 🚀
    """)


def render_synthesis_placeholder():
    """Placeholder Synthèse"""
    st.title("📊 Synthèse Comparative")

    st.info("""
    ### 🚧 Section en Construction

    Cette section proposera:

    - **Benchmarks comparatifs** de toutes les techniques
    - **Guide de sélection** : Quelle technique pour quel use case?
    - **Tableau récapitulatif** avec avantages/inconvénients
    - **Performances comparées** sur différents corpus
    - **Recommandations pratiques** pour la production

    **À venir prochainement!** 🚀
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # === SIDEBAR NAVIGATION ===
    with st.sidebar:
        st.title("🔍 Explorateur")
        st.caption("Recherche Textuelle")

        # Navigation principale
        section = st.radio(
            "📚 Navigation:",
            ["🏠 Accueil", "📊 TF-IDF", "🎯 BM25", "🧠 Embeddings", "📊 Synthèse"],
            key="main_nav"
        )

        st.divider()

        # Configuration globale (si pas sur accueil)
        if section != "🏠 Accueil":
            st.markdown("### ⚙️ Configuration")

            # Sélection dataset
            datasets_info = get_all_datasets_info()
            dataset_names = [info['name'] for info in datasets_info]
            dataset_labels = {
                'recettes': '🍝 Recettes',
                'films': '🎬 Films',
                'wikipedia': '📚 Wikipedia'
            }

            selected_dataset = st.selectbox(
                "Dataset:",
                dataset_names,
                format_func=lambda x: dataset_labels.get(x, x),
                key="dataset_select"
            )

            # Taille dataset
            use_extended = st.checkbox(
                "📦 Dataset étendu",
                value=False,
                help="Plus de documents pour tester performances",
                key="extended_check"
            )

            # Info dataset
            dataset_info = next(info for info in datasets_info if info['name'] == selected_dataset)
            extended_sizes = {'recettes': 80, 'films': 70, 'wikipedia': 220}
            estimated_docs = extended_sizes.get(selected_dataset, 30) if use_extended else dataset_info['nb_docs']

            st.info(f"📊 ~{estimated_docs} documents{' (étendu)' if use_extended else ''}")

            # Paramètres avancés
            with st.expander("🔧 Avancés"):
                remove_stopwords = st.checkbox("Supprimer stopwords", value=True, key="stopwords_check")
                show_intermediate = st.checkbox("Calculs intermédiaires", value=False, key="intermediate_check")

        st.divider()
        st.caption("💡 Explore les sections pour apprendre!")

    # === ROUTING ===

    if section == "🏠 Accueil":
        render_home()

    elif section in ["📊 TF-IDF", "🎯 BM25"]:
        # Charger le dataset
        with st.spinner("🔄 Chargement du dataset..."):
            start_load = time.time()
            dataset = load_cached_dataset(selected_dataset, extended=use_extended)
            load_time = time.time() - start_load

            documents_texts = [doc['text'] for doc in dataset]
            documents_titles = [doc['title'] for doc in dataset]
            documents_categories = [doc['category'] for doc in dataset]

        # Créer les engines
        if section == "📊 TF-IDF" or section == "🎯 BM25":
            with st.spinner("🧮 Préparation des moteurs de recherche..."):
                start_fit = time.time()
                tfidf_engine = create_tfidf_engine(documents_texts, remove_stopwords=remove_stopwords)
                fit_time = time.time() - start_fit

        # Render la section appropriée
        if section == "📊 TF-IDF":
            render_tfidf_section(
                dataset, documents_texts, documents_titles, documents_categories,
                tfidf_engine, remove_stopwords, show_intermediate, load_time, fit_time
            )

        elif section == "🎯 BM25":
            render_bm25_section(
                dataset, documents_texts, documents_titles, documents_categories,
                tfidf_engine, remove_stopwords
            )

    elif section == "🧠 Embeddings":
        render_embeddings_placeholder()

    elif section == "📊 Synthèse":
        render_synthesis_placeholder()

    # === FOOTER ===
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem 0;">
        <p>Créé avec ❤️ pour l'apprentissage de la recherche textuelle</p>
        <p style="font-size: 0.9rem;">📚 TF-IDF • 🎯 BM25 • 🧠 Embeddings (à venir)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

