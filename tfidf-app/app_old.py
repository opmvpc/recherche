"""
TF-IDF Explorer - Application Streamlit Éducative
Application pédagogique pour enseigner les techniques de recherche textuelle
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tfidf_engine import TFIDFEngine, preprocess_text, cosine_similarity
from datasets import load_dataset, get_all_datasets_info
from visualizations import (
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
    plot_vocabulary_stats
)


# Configuration de la page
st.set_page_config(
    page_title="TF-IDF Explorer 🔍",
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
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


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


def main():
    # Titre principal
    st.markdown('<h1 class="main-title">🔍 TF-IDF Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Apprends la recherche textuelle avec TF-IDF de manière interactive!</p>', unsafe_allow_html=True)

    # === SIDEBAR ===
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Sélection du dataset
        datasets_info = get_all_datasets_info()
        dataset_names = [info['name'] for info in datasets_info]
        dataset_labels = {
            'recettes': '🍝 Recettes de Cuisine',
            'films': '🎬 Synopsis de Films',
            'wikipedia': '📚 Articles Wikipédia'
        }

        selected_dataset = st.selectbox(
            "Choisis un dataset:",
            dataset_names,
            format_func=lambda x: dataset_labels.get(x, x)
        )

        # Taille du dataset
        use_extended = st.checkbox(
            "📦 Dataset étendu (100-200 docs)",
            value=False,
            help="Active pour tester les performances avec plus de documents"
        )

        # Infos sur le dataset
        dataset_info = next(info for info in datasets_info if info['name'] == selected_dataset)
        base_docs = dataset_info['nb_docs']

        # Estimations réalistes pour chaque dataset étendu
        extended_sizes = {
            'recettes': 80,
            'films': 70,
            'wikipedia': 220
        }
        estimated_docs = extended_sizes.get(selected_dataset, base_docs * 3) if use_extended else base_docs

        st.info(f"**{dataset_info['description']}**\n\n"
                f"📊 ~{estimated_docs} documents{' (étendu)' if use_extended else ''}\n\n"
                f"🏷️ Catégories: {', '.join(dataset_info['categories'])}")

        # Paramètres avancés
        with st.expander("🔧 Paramètres avancés"):
            remove_stopwords = st.checkbox("Supprimer les stopwords", value=True,
                                          help="Retire les mots courants comme 'le', 'la', 'de', etc.")

            show_intermediate = st.checkbox("Afficher les calculs intermédiaires", value=False,
                                           help="Montre les étapes de calcul détaillées")

        st.divider()
        st.caption("💡 **Astuce**: Explore les différents tabs pour comprendre TF-IDF étape par étape!")

    # Chargement du dataset
    with st.spinner("🔄 Chargement du dataset..."):
        import time
        start_load = time.time()
        dataset = load_cached_dataset(selected_dataset, extended=use_extended)
        load_time = time.time() - start_load

        documents_texts = [doc['text'] for doc in dataset]
        documents_titles = [doc['title'] for doc in dataset]
        documents_categories = [doc['category'] for doc in dataset]

    # Création du moteur TF-IDF
    with st.spinner("🧮 Entraînement du modèle TF-IDF..."):
        start_fit = time.time()
        engine = create_tfidf_engine(documents_texts, remove_stopwords=remove_stopwords)
        fit_time = time.time() - start_fit

    # === TABS PRINCIPAUX ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📖 Introduction",
        "🔢 Concepts TF-IDF",
        "🔍 Recherche Interactive",
        "📊 Exploration du Corpus",
        "🎓 Exemple Pas-à-Pas",
        "⚡ Performances"
    ])

    # === TAB 1: INTRODUCTION ===
    with tab1:
        st.header("📖 Le Problème de la Recherche Textuelle")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### 🤔 Pourquoi la recherche simple par mots-clés ne suffit pas?

            Imagine que tu cherches des documents sur le **"chat"** dans une bibliothèque numérique.

            **Approche naïve:** Compter combien de fois le mot "chat" apparaît dans chaque document.

            #### ❌ Problèmes de cette approche:

            1. **Documents longs favorisés injustement**
               - Doc A (20 mots): "chat" apparaît 2 fois → 10% du document
               - Doc B (200 mots): "chat" apparaît 3 fois → 1.5% du document
               - Pourtant Doc B aurait un score plus élevé! 🤦

            2. **Mots communs polluent les résultats**
               - Les mots comme "le", "la", "de", "et" apparaissent partout
               - Ils ne nous aident pas à trouver des documents pertinents
               - Ils créent du "bruit" dans les résultats

            3. **Pas de notion de "rareté"**
               - Un mot qui apparaît dans TOUS les documents n'est pas informatif
               - Un mot qui n'apparaît que dans quelques documents est très discriminant
            """)

        with col2:
            st.markdown("### 📊 Exemple Visuel")
            # Créer un exemple simple
            example_docs = [
                "Le chat noir joue avec le ballon rouge",
                "Le chien brun court dans le parc vert",
                "Le chat blanc dort sur le canapé bleu"
            ]

            st.code("\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(example_docs)]))

            st.markdown("""
            **Recherche naïve pour "le":**
            - Doc 1: 2 occurrences
            - Doc 2: 2 occurrences
            - Doc 3: 2 occurrences

            ➡️ Tous les docs sont équivalents! 🤷

            **Mais pour "chat":**
            - Doc 1: 1 occurrence
            - Doc 2: 0 occurrence
            - Doc 3: 1 occurrence

            ➡️ Plus informatif! 💡
            """)

        st.divider()

        st.markdown("""
        ### ✅ La Solution: TF-IDF

        **TF-IDF** (Term Frequency - Inverse Document Frequency) résout ces problèmes en combinant deux mesures:

        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>📈 TF (Term Frequency)</h4>
            <p><strong>Fréquence locale:</strong> À quel point ce mot est fréquent dans CE document?</p>
            <p>✅ Normalise par la longueur du document</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>📉 IDF (Inverse Document Frequency)</h4>
            <p><strong>Rareté globale:</strong> À quel point ce mot est rare dans TOUS les documents?</p>
            <p>✅ Pénalise les mots communs</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Formule Finale
        """)

        st.latex(r"\text{TF-IDF}(mot, doc) = \text{TF}(mot, doc) \times \text{IDF}(mot)")

        st.markdown("""
        **Résultat:** Un mot obtient un score élevé si:
        - ✅ Il est **fréquent dans le document** (TF élevé)
        - ✅ Il est **rare dans le corpus** (IDF élevé)

        ➡️ C'est exactement ce qu'on veut pour trouver des documents pertinents! 🎉
        """)

        st.info("💡 **Explore les autres tabs pour comprendre chaque composante en détail!**")

    # === TAB 2: CONCEPTS TF-IDF ===
    with tab2:
        st.header("🔢 Comprendre TF-IDF en Profondeur")

        # Term Frequency (TF)
        with st.expander("📈 **Term Frequency (TF)** - Fréquence des termes", expanded=True):
            st.markdown("""
            ### 💡 Intuition

            **"Si un mot apparaît souvent dans un document, le document parle probablement de ce sujet"**

            Mais attention! On doit **normaliser** par la longueur du document.
            """)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📐 Formule")
                st.latex(r"\text{TF}(mot, doc) = \frac{\text{nb d'occurrences du mot}}{\text{nb total de mots dans le doc}}")

                st.markdown("""
                **Exemple:**
                - Document: "Le chat noir. Le chat blanc."
                - Total: 6 mots
                - TF("chat") = 2/6 = 0.333
                - TF("le") = 2/6 = 0.333
                - TF("noir") = 1/6 = 0.167
                """)

            with col2:
                st.markdown("### 📊 Visualisation")
                # Prendre 3 documents exemples
                sample_indices = [0, 1, 2]
                sample_titles = [documents_titles[i] for i in sample_indices]

                fig_tf = plot_tf_comparison(engine.documents, sample_indices, sample_titles)
                st.pyplot(fig_tf)

            st.markdown("""
            ### 🎯 Pourquoi normaliser?

            Imagine deux documents:
            - **Doc A (10 mots):** Le mot "pizza" apparaît **2 fois**
            - **Doc B (100 mots):** Le mot "pizza" apparaît **3 fois**

            Sans normalisation, Doc B semble plus pertinent (3 > 2).

            **Mais!** Doc A consacre 20% de son contenu au mot "pizza" (2/10),
            tandis que Doc B seulement 3% (3/100).

            **Doc A est donc plus "à propos" de la pizza!** 🍕🎯
            """)

            st.warning("""
            ⚠️ **Problème de TF seul:** Les mots communs ("le", "la", "de")
            apparaissent fréquemment partout et polluent les résultats.

            ➡️ C'est là qu'intervient **IDF**!
            """)

        # Inverse Document Frequency (IDF)
        with st.expander("📉 **Inverse Document Frequency (IDF)** - Rareté des mots"):
            st.markdown("""
            ### 💡 Intuition

            **"Un mot rare est plus informatif qu'un mot commun"**

            Si un mot apparaît dans presque tous les documents, il ne nous aide pas à distinguer les documents entre eux.
            """)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📐 Formule")
                st.latex(r"\text{IDF}(mot) = \log\left(\frac{\text{nb total de docs}}{\text{nb de docs contenant le mot}}\right) + 1")

                st.markdown("""
                **Exemple avec 100 documents:**
                - Mot "le" apparaît dans 99 docs
                  - IDF = log(100/99) + 1 ≈ 1.01
                - Mot "pizza" apparaît dans 5 docs
                  - IDF = log(100/5) + 1 ≈ 4.00

                ➡️ "pizza" a un IDF **4x plus élevé** que "le"!
                """)

            with col2:
                st.markdown("### 📊 Visualisation")
                fig_idf = plot_idf_curve(engine.idf_vector, engine.vocabulary, engine.documents)
                st.pyplot(fig_idf)

            st.markdown("""
            ### 🤔 Pourquoi le logarithme?

            Le log **compresse** l'échelle pour éviter que les mots très rares dominent trop.

            **Sans log:**
            - Mot dans 100/100 docs: 100/100 = 1.0
            - Mot dans 1/100 docs: 100/1 = 100.0
            - Différence: **100x**

            **Avec log:**
            - Mot dans 100/100 docs: log(1) = 0
            - Mot dans 1/100 docs: log(100) ≈ 4.6
            - Différence: Plus raisonnable!
            """)

            st.markdown("### ☁️ Word Cloud par IDF")
            idf_dict = {engine.vocabulary[i]: engine.idf_vector[i]
                       for i in range(min(100, len(engine.vocabulary)))}
            fig_wc = plot_idf_wordcloud(idf_dict)
            st.pyplot(fig_wc)

            st.caption("Les mots les plus grands sont les plus rares (IDF élevé)")

        # TF-IDF Combiné
        with st.expander("🎯 **TF-IDF Combiné** - Le meilleur des deux mondes"):
            st.markdown("""
            ### 💡 Intuition

            **"Combine la fréquence locale ET la rareté globale"**

            Un mot obtient un score TF-IDF élevé s'il est:
            - ✅ Fréquent dans CE document (TF élevé)
            - ✅ Rare dans le corpus entier (IDF élevé)
            """)

            st.markdown("### 📐 Formule Finale")
            st.latex(r"\text{TF-IDF}(mot, doc) = \text{TF}(mot, doc) \times \text{IDF}(mot)")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("""
                ### 📋 Exemple Complet

                **Corpus de 3 documents:**
                1. "Le chat noir joue"
                2. "Le chien brun court"
                3. "Le chat blanc dort"

                **Pour le mot "chat" dans Doc 1:**

                **Étape 1: Calculer TF**
                - Occurrences: 1
                - Total mots: 4
                - TF = 1/4 = 0.25

                **Étape 2: Calculer IDF**
                - Docs avec "chat": 2 (Doc 1, Doc 3)
                - Total docs: 3
                - IDF = log(3/2) + 1 = 1.41

                **Étape 3: Multiplier**
                - TF-IDF = 0.25 × 1.41 = **0.35**
                """)

            with col2:
                st.markdown("### 📊 Heatmap TF-IDF")
                fig_heatmap = plot_tfidf_heatmap(
                    engine.tfidf_matrix,
                    engine.vocabulary,
                    documents_titles,
                    top_words=15
                )
                st.pyplot(fig_heatmap)

            st.success("""
            ✅ **Résultat:** Les cellules les plus rouges (scores élevés) représentent des mots:
            - Fréquents dans leur document
            - Rares dans le corpus
            - Donc très **discriminants**!
            """)

        # Cosine Similarity
        with st.expander("📐 **Cosine Similarity** - Mesurer la similarité"):
            st.markdown("""
            ### 💡 Intuition Géométrique

            **"Les documents sont des vecteurs dans un espace multidimensionnel"**

            Chaque dimension = un mot du vocabulaire

            La similarité cosinus mesure **l'angle** entre deux vecteurs:
            - Angle de 0° → similarité = 1 (identiques)
            - Angle de 90° → similarité = 0 (orthogonaux, rien en commun)
            """)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📐 Formule")
                st.latex(r"\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \times \|\vec{B}\|}")

                st.latex(r"= \frac{\sum_{i} A_i \times B_i}{\sqrt{\sum_i A_i^2} \times \sqrt{\sum_i B_i^2}}")

                st.markdown("""
                **Étapes de calcul:**

                1. **Dot product (produit scalaire):** Multiplier composante par composante
                2. **Norme de A:** √(somme des carrés de A)
                3. **Norme de B:** √(somme des carrés de B)
                4. **Diviser:** dot_product / (norme_A × norme_B)
                """)

            with col2:
                st.markdown("### 🎨 Visualisation 3D")
                fig_3d = plot_documents_3d(
                    engine.tfidf_matrix,
                    documents_titles,
                    documents_categories
                )
                st.plotly_chart(fig_3d, use_container_width=True)

                st.caption("Documents proches = contenu similaire")

            st.markdown("""
            ### 🤔 Pourquoi la similarité cosinus et pas juste additionner?

            **Problème sans normalisation:**
            - Doc A: [2, 0, 4] (long document)
            - Doc B: [1, 0, 2] (court document, même ratio)
            - Distance euclidienne suggère qu'ils sont différents!

            **Avec cosinus:**
            - On mesure l'**orientation** pas la **magnitude**
            - Docs avec le même ratio de mots = similaires
            - Indépendant de la longueur!
            """)

            st.info("💡 C'est cette mesure qu'on utilise pour la recherche!")

    # === TAB 3: RECHERCHE INTERACTIVE ===
    with tab3:
        st.header("🔍 Recherche Interactive")

        st.markdown("""
        Teste le système de recherche TF-IDF! Entre une requête et découvre les documents les plus pertinents.
        """)

        # Exemples de queries
        example_queries = {
            'recettes': ["plat italien", "cuisine épicée", "dessert chocolat", "poisson grillé"],
            'films': ["science-fiction espace", "comédie romantique", "super-héros action", "film horreur"],
            'wikipedia': ["guerre mondiale", "intelligence artificielle", "football champion", "physique quantique"]
        }

        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "🔎 Entre ta requête:",
                placeholder="Ex: recette italienne pâtes...",
                help="Entre des mots-clés décrivant ce que tu cherches"
            )

        with col2:
            st.markdown("**Ou choisis un exemple:**")
            if st.button("📝 Exemple 1"):
                query = example_queries[selected_dataset][0]
            if st.button("📝 Exemple 2"):
                query = example_queries[selected_dataset][1]

        top_k = st.slider("Nombre de résultats:", 3, 20, 5)

        if query and st.button("🚀 Rechercher!", type="primary"):
            with st.spinner("🔍 Recherche en cours..."):
                # Effectuer la recherche
                results = engine.search(query, top_k=top_k)

                if len(results) == 0 or all(score == 0 for _, score in results):
                    st.warning("😕 Aucun résultat pertinent trouvé. Essaie d'autres mots-clés!")
                else:
                    st.success(f"✅ Trouvé {len(results)} résultats!")

                    # Visualisation des scores
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 📊 Scores de similarité")
                        fig_results = plot_search_results(results, documents_titles, query)
                        st.pyplot(fig_results)

                    with col2:
                        st.markdown("### 🎯 Top résultats")
                        for rank, (doc_idx, score) in enumerate(results[:5], 1):
                            with st.container():
                                st.markdown(f"**#{rank} - Score: {score:.3f}**")
                                st.caption(f"Catégorie: {documents_categories[doc_idx]}")
                                st.markdown(f"*{documents_titles[doc_idx]}*")

                                if st.checkbox(f"Voir détails #{rank}", key=f"detail_{rank}"):
                                    st.text(documents_texts[doc_idx][:300] + "...")

                    st.divider()

                    # Afficher les calculs intermédiaires
                    if show_intermediate:
                        st.markdown("### 🔬 Calculs Détaillés (Premier Résultat)")

                        best_doc_idx = results[0][0]
                        explanation = engine.get_explanation(query, best_doc_idx)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### TF de la Query")
                            df_tf_query = pd.DataFrame(
                                list(explanation['tf_query'].items()),
                                columns=['Mot', 'TF']
                            ).sort_values('TF', ascending=False)
                            st.dataframe(df_tf_query, hide_index=True)

                        with col2:
                            st.markdown("#### IDF des mots")
                            df_idf = pd.DataFrame(
                                list(explanation['idf'].items()),
                                columns=['Mot', 'IDF']
                            ).sort_values('IDF', ascending=False)
                            st.dataframe(df_idf, hide_index=True)

                        with col3:
                            st.markdown("#### TF-IDF Final")
                            df_tfidf_query = pd.DataFrame(
                                list(explanation['tfidf_query'].items()),
                                columns=['Mot', 'TF-IDF']
                            ).sort_values('TF-IDF', ascending=False)
                            st.dataframe(df_tfidf_query, hide_index=True)

                        st.markdown("#### 📐 Calcul de Similarité Cosinus")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Dot Product", f"{explanation['dot_product']:.4f}")
                        col2.metric("Norme Query", f"{explanation['norm_query']:.4f}")
                        col3.metric("Norme Doc", f"{explanation['norm_doc']:.4f}")
                        col4.metric("Similarité", f"{explanation['cosine_similarity']:.4f}")

                        st.latex(r"\text{Similarité} = \frac{\text{Dot Product}}{\text{Norme Query} \times \text{Norme Doc}} = \frac{" +
                                f"{explanation['dot_product']:.4f}" + "}{" +
                                f"{explanation['norm_query']:.4f} \\times {explanation['norm_doc']:.4f}" +
                                "} = " + f"{explanation['cosine_similarity']:.4f}")

    # === TAB 4: EXPLORATION DU CORPUS ===
    with tab4:
        st.header("📊 Exploration du Corpus")

        # Statistiques générales
        st.markdown("### 📈 Statistiques Générales")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("📚 Documents", len(documents_texts))
        col2.metric("🔤 Vocabulaire", len(engine.vocabulary))
        col3.metric("📝 Mots/Doc (moy)", f"{np.mean([len(doc) for doc in engine.documents]):.1f}")
        col4.metric("🏷️ Catégories", len(set(documents_categories)))

        st.divider()

        # Visualisations du corpus
        tab_stats1, tab_stats2, tab_stats3 = st.tabs([
            "📊 Statistiques de base",
            "🗺️ Projection des documents",
            "🔥 Top mots par catégorie"
        ])

        with tab_stats1:
            st.markdown("### 📊 Distribution et Fréquences")

            fig_vocab = plot_vocabulary_stats(engine.documents)
            st.pyplot(fig_vocab)

            st.markdown("### 🌡️ Heatmap TF-IDF Complète")
            fig_heatmap_full = plot_tfidf_heatmap(
                engine.tfidf_matrix,
                engine.vocabulary,
                documents_titles,
                top_words=20
            )
            st.pyplot(fig_heatmap_full)

        with tab_stats2:
            st.markdown("### 🗺️ Projection des Documents dans l'Espace")

            viz_type = st.radio("Type de visualisation:", ["2D", "3D"], horizontal=True)

            if viz_type == "3D":
                fig_proj = plot_documents_3d(
                    engine.tfidf_matrix,
                    documents_titles,
                    documents_categories
                )
                st.plotly_chart(fig_proj, use_container_width=True)

                st.info("""
                💡 **Interprétation:**
                - Documents proches = vocabulaire similaire
                - Clusters de couleurs = catégories similaires
                - Utilise PCA pour réduire à 3 dimensions
                """)
            else:
                fig_proj = plot_documents_2d(
                    engine.tfidf_matrix,
                    documents_titles,
                    documents_categories
                )
                st.plotly_chart(fig_proj, use_container_width=True)

        with tab_stats3:
            st.markdown("### 🔥 Top Mots par Catégorie")

            selected_category = st.selectbox(
                "Choisis une catégorie:",
                sorted(set(documents_categories))
            )

            # Filtrer les docs de cette catégorie
            category_indices = [i for i, cat in enumerate(documents_categories) if cat == selected_category]

            # Calculer les mots moyens pour cette catégorie
            category_tfidf = engine.tfidf_matrix[category_indices].mean(axis=0)
            top_indices = np.argsort(category_tfidf)[-20:][::-1]

            top_words_cat = [(engine.vocabulary[i], category_tfidf[i]) for i in top_indices]

            # Visualisation
            words = [w for w, s in top_words_cat]
            scores = [s for w, s in top_words_cat]

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(words)), scores, color='#ff7f0e')
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Score TF-IDF moyen', fontweight='bold')
            ax.set_title(f'Top 20 mots pour: {selected_category}', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig)

            st.caption(f"🔍 Ces mots caractérisent la catégorie '{selected_category}'")

    # === TAB 5: EXEMPLE PAS-À-PAS ===
    with tab5:
        st.header("🎓 Exemple Complet Pas-à-Pas")

        st.markdown("""
        Suivons **tout le processus de calcul** étape par étape avec un exemple concret!
        """)

        # Sélectionner 3 documents
        st.markdown("### 📚 Étape 0: Sélection des Documents")

        sample_size = 3
        sample_indices = list(range(min(sample_size, len(documents_texts))))

        for idx in sample_indices:
            with st.expander(f"📄 Document {idx+1}: {documents_titles[idx]}"):
                st.write(documents_texts[idx])
                st.caption(f"Catégorie: {documents_categories[idx]}")

        # Query prédéfinie
        example_query = example_queries[selected_dataset][0]
        tutorial_query = st.text_input("🔎 Query pour l'exemple:", value=example_query)

        if tutorial_query:
            st.markdown("### 🔢 Étape 1: Calcul des TF (Term Frequency)")

            # Créer un mini engine avec seulement ces 3 docs
            sample_texts = [documents_texts[i] for i in sample_indices]
            mini_engine = TFIDFEngine(sample_texts, remove_stopwords=remove_stopwords)
            mini_engine.fit()

            # TF pour chaque document
            tf_data = []
            for doc_idx in range(len(sample_indices)):
                tf_dict = mini_engine.compute_tf(doc_idx)
                for word, tf_val in tf_dict.items():
                    tf_data.append({
                        'Document': f"Doc {doc_idx+1}",
                        'Mot': word,
                        'TF': round(tf_val, 4)
                    })

            df_tf = pd.DataFrame(tf_data)
            df_tf_pivot = df_tf.pivot(index='Mot', columns='Document', values='TF').fillna(0)

            st.dataframe(df_tf_pivot, use_container_width=True)
            st.caption("💡 TF = nombre d'occurrences / total mots du document")

            st.markdown("### 🔢 Étape 2: Calcul des IDF (Inverse Document Frequency)")

            idf_dict = mini_engine.compute_idf()

            # Calculer le nombre de documents contenant chaque mot
            doc_counts = {}
            for word in mini_engine.vocabulary:
                count = sum(1 for doc in mini_engine.documents if word in doc)
                doc_counts[word] = count

            idf_data = []
            for word, idf_val in idf_dict.items():
                idf_data.append({
                    'Mot': word,
                    'Nb Docs': doc_counts[word],
                    'IDF': round(idf_val, 4)
                })

            df_idf = pd.DataFrame(idf_data).sort_values('IDF', ascending=False)

            st.dataframe(df_idf.head(20), hide_index=True, use_container_width=True)
            st.caption("💡 IDF = log(total docs / nb docs contenant le mot) + 1")

            st.markdown("### 🔢 Étape 3: Calcul des TF-IDF (multiplication)")

            st.latex(r"\text{TF-IDF} = \text{TF} \times \text{IDF}")

            # Créer la matrice TF-IDF
            tfidf_data = []
            for doc_idx in range(len(sample_indices)):
                for word_idx, word in enumerate(mini_engine.vocabulary):
                    tf_val = mini_engine.tf_matrix[doc_idx, word_idx]
                    idf_val = mini_engine.idf_vector[word_idx]
                    tfidf_val = tf_val * idf_val

                    if tfidf_val > 0:
                        tfidf_data.append({
                            'Document': f"Doc {doc_idx+1}",
                            'Mot': word,
                            'TF': round(tf_val, 4),
                            'IDF': round(idf_val, 4),
                            'TF-IDF': round(tfidf_val, 4)
                        })

            df_tfidf = pd.DataFrame(tfidf_data).sort_values('TF-IDF', ascending=False)

            st.dataframe(df_tfidf.head(30), hide_index=True, use_container_width=True)

            st.markdown("### 🔢 Étape 4: Vectorisation de la Query")

            query_vector = mini_engine.vectorize_query(tutorial_query)
            query_tokens = preprocess_text(tutorial_query, remove_stopwords)

            st.write(f"**Query:** {tutorial_query}")
            st.write(f"**Tokens:** {query_tokens}")

            # Afficher le vecteur de la query
            query_tfidf = {}
            for word_idx, word in enumerate(mini_engine.vocabulary):
                if query_vector[word_idx] > 0:
                    query_tfidf[word] = query_vector[word_idx]

            df_query = pd.DataFrame(
                list(query_tfidf.items()),
                columns=['Mot', 'TF-IDF']
            ).sort_values('TF-IDF', ascending=False)

            st.dataframe(df_query, hide_index=True)

            st.markdown("### 🔢 Étape 5: Calcul de la Similarité Cosinus")

            results = mini_engine.search(tutorial_query, top_k=3)

            for rank, (doc_idx, score) in enumerate(results, 1):
                with st.expander(f"📄 #{rank} - {documents_titles[sample_indices[doc_idx]]} (Score: {score:.4f})"):
                    explanation = mini_engine.get_explanation(tutorial_query, doc_idx)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Calculs:**")
                        st.write(f"Dot Product: {explanation['dot_product']:.6f}")
                        st.write(f"Norme Query: {explanation['norm_query']:.6f}")
                        st.write(f"Norme Document: {explanation['norm_doc']:.6f}")

                    with col2:
                        st.markdown("**Formule:**")
                        st.latex(r"\cos(\theta) = \frac{" +
                                f"{explanation['dot_product']:.4f}" + "}{" +
                                f"{explanation['norm_query']:.4f} \\times {explanation['norm_doc']:.4f}" +
                                "}")
                        st.latex(r"= " + f"{explanation['cosine_similarity']:.4f}")

            st.success("🎉 **Terminé!** Tu as suivi tout le processus de A à Z!")

    # === TAB 6: PERFORMANCES ===
    with tab6:
        st.header("⚡ Analyse des Performances de TF-IDF")

        st.markdown("""
        Cette section explique la **complexité algorithmique** de TF-IDF et comment les performances
        évoluent avec la taille du corpus. Essentiel pour comprendre les limites pratiques!
        """)

        # Métriques actuelles
        st.markdown("### 📈 Performances du Corpus Actuel")

        n_docs = len(documents_texts)
        n_vocab = len(engine.vocabulary)
        avg_doc_length = np.mean([len(doc) for doc in engine.documents])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("⏱️ Chargement", f"{load_time:.3f}s")
        col2.metric("🧮 Entraînement", f"{fit_time:.3f}s")
        col3.metric("📚 Documents", n_docs)
        col4.metric("🔤 Vocabulaire", n_vocab)

        st.divider()

        # Complexité algorithmique
        st.markdown("### 🧮 Complexité Algorithmique")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            #### 📐 Notations

            - **n** = nombre de documents
            - **m** = taille moyenne d'un document (en mots)
            - **v** = taille du vocabulaire

            #### ⏱️ Complexité Temporelle

            **1. Construction du vocabulaire:** `O(n × m)`
            - Parcourir tous les mots de tous les documents
            - Ajouter chaque mot unique au vocabulaire

            **2. Calcul de la matrice TF:** `O(n × m)`
            - Pour chaque document: compter les occurrences
            - Normaliser par la longueur

            **3. Calcul du vecteur IDF:** `O(n × v)`
            - Pour chaque mot du vocabulaire
            - Compter dans combien de documents il apparaît

            **4. Calcul de TF-IDF:** `O(n × v)`
            - Multiplication élément par élément

            **5. Recherche (similarité cosinus):** `O(n × v)`
            - Calculer la similarité entre query et chaque doc

            ---

            **Complexité totale: `O(n × m + n × v)`**

            En pratique, **v << n × m** (vocabulaire petit vs corpus total)

            ➡️ Dominé par **`O(n × v)`** après la phase de construction
            """)

        with col2:
            st.markdown("""
            #### 💾 Complexité Spatiale

            **Matrice TF-IDF:** `O(n × v)`

            Pour notre corpus:
            """)

            matrix_size_mb = (n_docs * n_vocab * 8) / (1024 * 1024)  # float64 = 8 bytes
            st.metric("Taille Matrice", f"{matrix_size_mb:.2f} MB")

            st.markdown("""
            **Problème:** Pour de gros corpus (millions de docs),
            la matrice devient **énorme**!

            **Solution:** Matrice **creuse** (sparse matrix)
            - Stocke seulement les valeurs non-nulles
            - Typiquement 95-99% de zéros dans TF-IDF
            """)

            # Calcul de la sparsité
            non_zero = np.count_nonzero(engine.tfidf_matrix)
            total_elements = n_docs * n_vocab
            sparsity = 100 * (1 - non_zero / total_elements)

            st.metric("Sparsité", f"{sparsity:.1f}%")
            st.caption(f"Seulement {100-sparsity:.1f}% de valeurs non-nulles!")

        st.divider()

        # Benchmark avec différentes tailles
        st.markdown("### 📊 Simulation de Performance selon la Taille")

        st.markdown("""
        Voici comment les temps de calcul évoluent théoriquement avec la taille du corpus:
        """)

        # Simuler des temps pour différentes tailles
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]

        # Estimer les temps basés sur le temps actuel
        base_time_per_doc = fit_time / n_docs if n_docs > 0 else 0.001

        estimated_times = []
        estimated_vocab = []
        estimated_memory = []

        for size in sizes:
            # Temps croît linéairement avec n
            est_time = base_time_per_doc * size
            estimated_times.append(est_time)

            # Vocabulaire croît en log(n) approximativement
            est_vocab = int(n_vocab * np.log(1 + size) / np.log(1 + n_docs))
            estimated_vocab.append(est_vocab)

            # Mémoire = n × v × 8 bytes
            est_mem = (size * est_vocab * 8) / (1024 * 1024)
            estimated_memory.append(est_mem)

        df_perf = pd.DataFrame({
            'Nombre de Documents': sizes,
            'Temps (s)': [f"{t:.2f}" for t in estimated_times],
            'Vocabulaire': estimated_vocab,
            'Mémoire (MB)': [f"{m:.1f}" for m in estimated_memory]
        })

        st.dataframe(df_perf, hide_index=True, use_container_width=True)

        # Graphique de scaling
        col1, col2 = st.columns(2)

        with col1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sizes, estimated_times, marker='o', linewidth=2, markersize=8, color='#ff7f0e')
            ax.set_xlabel('Nombre de Documents', fontweight='bold', fontsize=12)
            ax.set_ylabel('Temps d\'entraînement (s)', fontweight='bold', fontsize=12)
            ax.set_title('Scaling du Temps de Calcul', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sizes, estimated_memory, marker='s', linewidth=2, markersize=8, color='#d62728')
            ax.set_xlabel('Nombre de Documents', fontweight='bold', fontsize=12)
            ax.set_ylabel('Mémoire (MB)', fontweight='bold', fontsize=12)
            ax.set_title('Consommation Mémoire', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.tight_layout()
            st.pyplot(fig)

        st.divider()

        # Optimisations possibles
        st.markdown("### 🚀 Optimisations Possibles")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### ✅ Optimisations Implémentables

            **1. Matrice Creuse (Sparse Matrix)**
            ```python
            from scipy.sparse import csr_matrix
            tfidf_sparse = csr_matrix(tfidf_matrix)
            # Économie de 95%+ de mémoire!
            ```

            **2. Min/Max Document Frequency**
            ```python
            # Ignorer mots trop rares ou trop communs
            min_df = 2  # Apparaît dans au moins 2 docs
            max_df = 0.8  # Apparaît dans max 80% des docs
            ```

            **3. Limitation du Vocabulaire**
            ```python
            # Garder seulement top N mots
            max_features = 5000
            ```

            **4. Indexation Inverse**
            - Stocker: mot → liste de docs contenant ce mot
            - Plus rapide pour la recherche
            - C'est ce que fait Elasticsearch!
            """)

        with col2:
            st.markdown("""
            #### 🔥 Alternatives Plus Rapides

            **1. BM25 (Best Matching 25)**
            - Variante améliorée de TF-IDF
            - Meilleurs résultats en pratique
            - Saturation du TF (évite over-counting)

            **2. Hashing Vectorizer**
            - Pas de vocabulaire explicite
            - Hash chaque mot → index
            - Très rapide, perd un peu de précision

            **3. Word Embeddings**
            - Word2Vec, GloVe, FastText
            - Capture la sémantique
            - Nécessite pré-entraînement

            **4. Transformers (BERT, etc.)**
            - État de l'art en NLP
            - Très lent à entraîner
            - Modèles pré-entraînés disponibles
            """)

        st.divider()

        # Comparaison avec sklearn
        st.markdown("### 🔬 Comparaison avec scikit-learn")

        st.markdown("""
        Notre implémentation pédagogique vs TfidfVectorizer optimisé de sklearn:
        """)

        col1, col2, col3 = st.columns(3)

        col1.markdown("""
        **Notre Implémentation** 🎓
        - ✅ Pédagogique et claire
        - ✅ Tous les états intermédiaires
        - ✅ Facile à debugger
        - ❌ Pas optimisée
        - ❌ Matrice dense
        """)

        col2.markdown("""
        **scikit-learn** ⚡
        - ✅ Très optimisée (C/Cython)
        - ✅ Matrice sparse
        - ✅ Production-ready
        - ❌ Boîte noire
        - ❌ Moins pédagogique
        """)

        col3.metric("Différence vitesse", "~10-50x plus rapide")
        col3.caption("sklearn est beaucoup plus rapide, mais notre code est plus éducatif!")

        # Benchmark optionnel
        if st.checkbox("🧪 Faire un benchmark avec sklearn?"):
            with st.spinner("Benchmarking en cours..."):
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Notre implémentation
                start = time.time()
                custom_engine = TFIDFEngine(documents_texts[:100], remove_stopwords=remove_stopwords)
                custom_engine.fit()
                custom_time = time.time() - start

                # sklearn
                start = time.time()
                vectorizer = TfidfVectorizer(max_features=None)
                vectorizer.fit_transform(documents_texts[:100])
                sklearn_time = time.time() - start

                col1, col2, col3 = st.columns(3)
                col1.metric("Notre implémentation", f"{custom_time:.4f}s")
                col2.metric("scikit-learn", f"{sklearn_time:.4f}s")
                col3.metric("Speedup", f"{custom_time/sklearn_time:.1f}x")

                st.info(f"💡 sklearn est **{custom_time/sklearn_time:.1f}x plus rapide** grâce à ses optimisations C/Cython!")

        st.divider()

        # Conseils pratiques
        st.markdown("### 💡 Conseils Pratiques")

        st.success("""
        **Pour un Projet Réel:**

        1. **Petit corpus (<10k docs):** Notre implémentation suffit amplement
        2. **Corpus moyen (10k-100k):** Utiliser sklearn avec matrice sparse
        3. **Gros corpus (>100k):** Considérer Elasticsearch ou Solr
        4. **Corpus énorme (>1M):** Utiliser une base de données spécialisée (Elasticsearch, Milvus, etc.)

        **Pour la Recherche Sémantique:**
        - TF-IDF: Mots-clés exacts
        - Embeddings: Sens et synonymes
        - Hybrid: Combiner les deux!
        """)

        st.warning("""
        ⚠️ **Limites de TF-IDF:**
        - Ne comprend pas les **synonymes** ("voiture" ≠ "automobile")
        - Ne comprend pas le **contexte** ("banque" = finance ou rivière?)
        - Vocabulaire doit être **exact** (typos = échec)

        ➡️ Les embeddings et transformers résolvent ces problèmes!
        """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Créé avec ❤️ pour l'apprentissage de TF-IDF</p>
        <p style="font-size: 0.9rem;">📚 Application éducative - Streamlit + Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
