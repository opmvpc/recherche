"""
Sections Embeddings et Synthèse pour l'application
À intégrer dans app.py principal
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

# Imports des visualizations nécessaires
from src.visualizations import (
    plot_embedding_space_3d,
    plot_clustering_2d,
    plot_multi_technique_comparison,
    plot_hybrid_alpha_effect,
)


# ============================================================================
# CACHE FUNCTIONS POUR EMBEDDINGS
# ============================================================================


@st.cache_resource
def create_embedding_engine(
    documents_texts: list, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):
    """Crée et index le moteur embeddings avec cache"""
    from src.embedding_engine import EmbeddingSearch

    engine = EmbeddingSearch(model_name=model_name)
    with st.spinner("🧠 Calcul des embeddings (peut prendre 1-2min)..."):
        engine.index(documents_texts, use_cache=True, show_progress=False)
    return engine


def create_hybrid_engine(
    documents_texts: list, bm25_engine, embedding_engine, alpha: float = 0.5
):
    """
    Crée le moteur hybrid (pas de cache car très rapide)
    """
    from src.hybrid_search import HybridSearch

    return HybridSearch(documents_texts, bm25_engine, embedding_engine, alpha=alpha)


# ============================================================================
# SECTION EMBEDDINGS COMPLÈTE
# ============================================================================


def render_embeddings_section(
    dataset,
    documents_texts,
    documents_titles,
    documents_categories,
    tfidf_engine,
    bm25_engine,
    remove_stopwords,
    embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Section Embeddings complète avec tous les onglets

    Args:
        embedding_model_name: Nom du modèle HuggingFace à utiliser (sélectionné dans sidebar)
    """

    st.title("🧠 Embeddings Vectoriels: Recherche Sémantique")

    # Afficher le modèle sélectionné
    model_label = embedding_model_name.split("/")[-1]  # Prendre juste le nom
    st.caption(f"🤖 Modèle: **{model_label}** | 💾 Chargé depuis le cache")

    # Import de la fonction de navigation stylée
    from app import render_tab_navigation

    # Sub-navigation avec beaux boutons
    tabs_list = [
        "📖 Introduction",
        "🔢 Concepts",
        "🔍 Recherche",
        "📊 Exploration",
        "🎓 Pas-à-Pas",
        "⚔️ Comparaison",
        "🎨 Hybrid",
        "⚡ Performance",
    ]
    tab = render_tab_navigation(
        tabs_list, "embeddings_tabs", default_tab="📖 Introduction"
    )

    # Créer l'engine embeddings (avec cache - 1 SEUL téléchargement par modèle!)
    embedding_engine = create_embedding_engine(documents_texts, embedding_model_name)

    if tab == "📖 Introduction":
        render_embeddings_intro(documents_texts, tfidf_engine)
    elif tab == "🔢 Concepts":
        render_embeddings_concepts(embedding_engine, documents_texts)
    elif tab == "🔍 Recherche":
        render_embeddings_search(
            embedding_engine, documents_texts, documents_titles, documents_categories
        )
    elif tab == "📊 Exploration":
        render_embeddings_exploration(
            embedding_engine, documents_texts, documents_titles, documents_categories
        )
    elif tab == "🎓 Pas-à-Pas":
        render_embeddings_stepbystep(
            embedding_engine, documents_texts, documents_titles
        )
    elif tab == "⚔️ Comparaison":
        render_embeddings_comparison(
            embedding_engine,
            tfidf_engine,
            bm25_engine,
            documents_texts,
            documents_titles,
        )
    elif tab == "🎨 Hybrid":
        render_embeddings_hybrid(
            embedding_engine,
            bm25_engine,
            documents_texts,
            documents_titles,
            documents_categories,
        )
    elif tab == "⚡ Performance":
        render_embeddings_performance(
            embedding_engine, documents_texts, tfidf_engine, bm25_engine
        )


def render_embeddings_intro(documents_texts, tfidf_engine):
    """Introduction & Limites des approches lexicales"""
    st.header("📖 Au-delà des Mots: La Recherche Sémantique")

    st.info("""
    📊 **Récapitulatif de votre parcours d'apprentissage:**

    - **TF-IDF (1970s):** Recherche par fréquence des mots, pondérée par rareté
    - **BM25 (1994):** TF-IDF amélioré avec saturation et normalisation intelligente

    **Principe commun:** Recherche **LEXICALE** = matching de mots exacts (comptage de tokens)

    **Limite fondamentale:** Ces algorithmes ne comprennent PAS le sens des mots! 🤯
    """)

    st.divider()

    st.markdown("### ❌ Les 4 Fails Critiques des Approches Lexicales")

    # Fail #1: Synonymes
    st.error("""
    **Fail #1: Synonymes Ignorés 😵**

    TF-IDF et BM25 ne comprennent PAS que des mots différents peuvent avoir le même sens!
    """)

    st.markdown("""
    **Exemple Concret:**

    Imaginons que tu cherches des infos sur les **voitures rapides**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔎 Ta Query:**")
        st.code("voiture rapide", language="text")

    with col2:
        st.markdown("**📄 Un Document Pertinent:**")
        st.code("automobile véloce", language="text")

    st.markdown("""
    **Analyse lexicale (TF-IDF/BM25):**
    - Mots query: `["voiture", "rapide"]`
    - Mots doc: `["automobile", "véloce"]`
    - **Intersection:** ∅ (vide!)

    **Verdict lexical:** Aucun mot commun → Score = 0.00 😭

    **Analyse sémantique (Embeddings):**
    - "voiture" ≈ "automobile" (synonymes)
    - "rapide" ≈ "véloce" (synonymes)

    **Verdict sémantique:** Sens identique → Score = 0.94 🔥
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Score TF-IDF/BM25",
            "0.00",
            delta="Aucun mot commun!",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "Score Embeddings", "0.94", delta="Sens identique!", delta_color="normal"
        )

    st.success("""
    ✅ **Pourquoi Embeddings gagne:**

    Les embeddings capturent la **proximité sémantique** entre mots.
    Dans l'espace vectoriel, "voiture" et "automobile" sont des vecteurs **TRÈS PROCHES**,
    parce que le modèle a appris qu'ils apparaissent dans des contextes similaires!
    """)

    st.divider()

    # Fail #2: Polysémie
    st.error("""
    **Fail #2: Polysémie (Mots à Double Sens) 🍎💻**

    Un même mot peut avoir des sens **DIFFÉRENTS** selon le contexte!
    """)

    st.markdown("""
    **Définition:**
    **Polysémie** = Un mot avec plusieurs significations

    **Exemple classique avec "Apple":**
    """)

    poly_example = pd.DataFrame(
        {
            "Document": [
                "Apple fait de bons ordinateurs et smartphones",
                "Apple est un fruit délicieux et sain",
            ],
            "Sens Réel": ["💻 Entreprise tech", "🍎 Fruit"],
            'Score TF-IDF (query: "Apple")': ["0.87", "0.87"],
            "Score Embeddings": ["Différenciés!", "Selon contexte"],
        }
    )

    st.dataframe(poly_example, use_container_width=True)

    st.markdown("""
    **Problème avec TF-IDF/BM25:**
    - Si ta query est "Apple ordinateur", les DEUX docs matchent "Apple" également
    - Impossible de distinguer le sens! 😵

    **Solution Embeddings:**
    - Contexte 1: `["Apple", "ordinateurs", "smartphones"]` → Vecteur orienté "tech"
    - Contexte 2: `["Apple", "fruit", "délicieux"]` → Vecteur orienté "nourriture"

    Les embeddings capturent le **contexte global** et génèrent des vecteurs différents!

    **Exemple en français:**
    - "La **banque** est fermée" → 🏦 Institution financière
    - "La **banque** du fleuve" → 🏞️ Bord de rivière
    """)

    st.info("""
    💡 **Comment ça marche?**

    Le mécanisme d'**Attention** (dans les Transformers) regarde les mots voisins:
    - "banque" + "fermée" → Probabilité institution financière: 95%
    - "banque" + "fleuve" → Probabilité bord de rivière: 92%

    Résultat: **Embeddings différents** selon le contexte! ✨
    """)

    st.divider()

    # Fail #3: Relations Conceptuelles
    st.error("""
    **Fail #3: Relations Conceptuelles Manquées 🗼🇫🇷**

    Incapable de comprendre les **relations implicites** entre concepts!
    """)

    st.markdown("""
    **Exemple: Connaissance Géographique**

    **Query:** "capitale France"
    """)

    example_data = pd.DataFrame(
        {
            "Document": [
                "Paris est une belle ville avec la Tour Eiffel",
                "La France est un grand pays européen",
            ],
            "Mots Communs (Query)": ["Aucun", '"France"'],
            "Score TF-IDF/BM25": [0.00, 0.73],
            "Score Embeddings": [0.88, 0.42],
            "Pertinence Réelle": ["✅ TRÈS pertinent!", "⚠️ Peu pertinent"],
        }
    )

    st.dataframe(example_data, use_container_width=True)

    st.markdown("""
    **Analyse du Fail:**

    **TF-IDF/BM25:**
    - Doc 1 ("Paris...") → Score = 0.00 (aucun mot commun!)
    - Doc 2 ("France...") → Score = 0.73 (matche "France")
    - **Classement:** Doc 2 > Doc 1

    **Mais en réalité:** Doc 1 est BEAUCOUP plus pertinent! 😱

    **Embeddings Comprend:**
    - "Paris" **EST LA** capitale de la France (relation sémantique)
    - "capitale" + "France" → Proximité avec "Paris" dans l'espace vectoriel
    - **Classement:** Doc 1 > Doc 2 ✅

    **Autres exemples de relations:**
    - "Picasso" ↔ "peinture cubisme"
    - "Einstein" ↔ "relativité physique"
    - "Mozart" ↔ "compositeur classique"
    """)

    st.warning("""
    ⚠️ **Limite importante:**

    Ces relations NE SONT PAS programmées! Elles sont **apprises** automatiquement
    pendant l'entraînement sur des milliards de phrases.

    Si le modèle n'a jamais vu certaines relations, il ne les connaîtra pas.
    """)

    st.divider()

    # Fail #4: Paraphrases
    st.error("""
    **Fail #4: Paraphrases Non Reconnues 🐱🐭**

    Deux phrases avec le **même sens** mais des **mots totalement différents**!
    """)

    phrase_a = "Le chat poursuit la souris"
    phrase_b = "Le félin traque le rongeur"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Phrase A:**")
        st.info(phrase_a)
        st.caption("Mots: `['chat', 'poursuit', 'souris']`")
    with col2:
        st.markdown("**Phrase B:**")
        st.info(phrase_b)
        st.caption("Mots: `['félin', 'traque', 'rongeur']`")

    st.markdown("""
    **Analyse Lexicale (comptage de mots):**
    - **Mots totaux:** 6 (3 par phrase)
    - **Mots communs:** 2 ("le" ×2) - Seulement les articles!
    - **Mots de contenu communs:** 0 😱
    - **Vocabulaire overlap:** 33% (2/6)

    **Scores TF-IDF/BM25:**
    - Similarité basée uniquement sur "le" (stopword!)
    - Score résultant: ~0.15 (très faible)
    - **Conclusion lexicale:** Documents NON similaires ❌
    """)

    similarity_comparison = pd.DataFrame(
        {
            "Méthode": ["TF-IDF/BM25 (lexical)", "Embeddings (sémantique)"],
            "Mots Matchés": ["2/6 (articles)", "Sens global"],
            "Score": [0.15, 0.91],
            "Verdict": ["❌ Non similaires", "✅ TRÈS similaires!"],
        }
    )

    st.dataframe(similarity_comparison, use_container_width=True)

    st.success("""
    ✅ **Pourquoi Embeddings comprend:**

    Le modèle a appris pendant son entraînement:
    - "chat" ≈ "félin" (relation animal/catégorie)
    - "poursuit" ≈ "traque" (synonyme d'action)
    - "souris" ≈ "rongeur" (relation animal/catégorie)

    **Résultat:** Même si AUCUN mot de contenu n'est identique,
    le **sens global** est capturé! 🎯
    """)

    st.markdown("""
    **Cas d'usage réels:**
    - **Customer support:** Détecter questions similaires formulées différemment
    - **Recherche académique:** Trouver papers sur le même sujet avec terminologie variée
    - **E-commerce:** Comprendre intentions d'achat malgré descriptions différentes
    """)

    st.divider()

    st.success("""
    ### ✅ La Solution: Embeddings Vectoriels

    **Révolution Paradigm:**

    Au lieu de **compter des mots** (approche symbolique),
    on **capture le SENS** dans un espace vectoriel dense (approche géométrique)!

    **Pipeline Simplifié:**
    ```
    Texte Brut
        ↓
    Transformer Neural Network (BERT/Sentence-BERT)
        ↓
    Vecteur Dense (384 dimensions)
        ↓
    Comparaison Géométrique (distance/angle)
        ↓
    Score de Similarité Sémantique
    ```

    **Le Magic Trick:** 🪄
    - "voiture" → `[0.23, -0.81, 0.45, ...]`
    - "automobile" → `[0.21, -0.79, 0.47, ...]`

    Ces deux vecteurs sont **PROCHES** dans l'espace à 384 dimensions!

    **Distance cosinus:** ~0.02 (très faible) → Sens similaire! ✅
    """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 Passons à la Suite!

    Dans les prochains onglets, tu vas apprendre:
    1. **Concepts:** Comment fonctionnent les Transformers et l'Attention
    2. **Recherche:** Tester la recherche sémantique interactive
    3. **Exploration:** Visualiser l'espace vectoriel en 3D
    4. **Pas-à-Pas:** Calculs détaillés d'un exemple complet
    5. **Comparaison:** Embeddings vs TF-IDF vs BM25
    6. **Hybrid:** Combiner le meilleur des deux mondes
    7. **Performance:** Optimisations et benchmarks

    Let's go! 🔥
    """)


def render_embeddings_concepts(embedding_engine, documents_texts):
    """Concepts détaillés des embeddings"""
    st.header("🔢 Comprendre les Embeddings en Profondeur")

    with st.expander("📊 **Sparse vs Dense: La Révolution**", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 TF-IDF (Sparse)")
            st.code("""
Vocabulaire: 10,000 mots
Doc: "Le chat mange"

Vector: [0, 0, 0, ..., 0.5, 0, ..., 0.8, ...]
        └──────────┬──────────┘
           99.97% de zéros!

Dimensions: 10,000
Non-zéros: ~3 (0.03%)
            """)
            st.warning("**Problème:** Énorme mais vide!")

        with col2:
            st.markdown("### 🧠 Embeddings (Dense)")
            st.code("""
Doc: "Le chat mange"

Vector: [0.234, -0.891, 0.456, ..., -0.123]
        └────────────┬────────────┘
        Toutes valeurs non-nulles!

Dimensions: 384
Non-zéros: 384 (100%)
            """)
            st.success("**Avantage:** Compact et riche!")

        st.info("""
        **💡 Pourquoi Dense est Mieux:**
        - Chaque dimension capture un "concept" sémantique
        - Pas de dimensions gaspillées
        - Représentation BEAUCOUP plus riche! ✨
        """)

    with st.expander("🔄 **Pipeline: De Texte à Vecteur**"):
        st.markdown("""
        ### Le Parcours d'un Texte dans le Réseau

        ```text
        1. Texte Brut
           "Le chat mange du poisson"

        2. Tokenization
           ["le", "chat", "mange", "du", "poisson"]

        3. Neural Network (Transformer - BERT)
           - Embedding Layer: mots → vecteurs initiaux
           - Attention Layers (×12): capture le contexte
           - Pooling: agrégation en UN seul vecteur

        4. Vecteur Final
           [0.234, -0.891, 0.456, ..., -0.123]  (384 dimensions)
        ```
        """)

        st.success(f"""
        **Votre modèle actuel:** `{embedding_engine.model_name}`

        - **Dimensions:** {embedding_engine.embedding_dim}
        - **Type:** Sentence-BERT multilingue
        - **Entraîné sur:** Des milliards de phrases
        """)

    with st.expander("🤔 **Qu'est-ce qu'un Transformer?**"):
        st.markdown("""
        ### Architecture BERT/Sentence-BERT

        **Transformer** = Architecture de réseau de neurones révolutionnaire (2017)
        - Utilisée par GPT, BERT, ChatGPT, etc.
        - Basée sur le mécanisme d'**Attention**

        ---

        ### 💡 Le Mécanisme d'Attention (le Cœur)

        **Problème à résoudre:** Comment comprendre qu'un mot a des sens différents selon le contexte?

        **Exemple classique:** Le mot **"banque"**
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Phrase 1:**
            "La **banque** est fermée"

            🏦 **Contexte:** institution financière

            **Mots clés:**
            - "fermée" (horaires)
            - Pas de fleuve/rivière
            """)

        with col2:
            st.success("""
            **Phrase 2:**
            "La **banque** du fleuve"

            🏞️ **Contexte:** bord de rivière

            **Mots clés:**
            - "fleuve" (géographie)
            - Pas d'horaires/argent
            """)

        st.markdown("""
        ### 🔍 Comment l'Attention Fonctionne

        **Mécanisme:** Chaque mot "regarde" tous les autres mots pour comprendre son sens!

        **Exemple avec la phrase:** "Le chat noir mange du poisson"
        """)

        # Tableau d'attention
        attention_example = pd.DataFrame(
            {
                "Mot": ["noir"],
                '→ "le"': ["0.05 (faible)"],
                '→ "chat"': ["0.75 (FORT!)"],
                '→ "noir"': ["0.02 (self)"],
                '→ "mange"': ["0.08 (faible)"],
                '→ "du"': ["0.03 (faible)"],
                '→ "poisson"': ["0.07 (faible)"],
            }
        )

        st.dataframe(attention_example, use_container_width=True, hide_index=True)

        st.markdown("""
        **Interprétation:**
        - "noir" regarde surtout vers "chat" (0.75) → Il décrit le chat!
        - Les autres mots ont peu d'attention → Moins importants pour comprendre "noir"

        **Ce que le réseau apprend:**
        - "noir" est un **adjectif** qui qualifie "chat"
        - Donc le vecteur de "noir" sera influencé par "chat"
        - Résultat: embeddings contextuels! 🎯

        ---

        ### 🏗️ Architecture Complète (Simplifié)

        ```
        Input: "Le chat noir"

        1. Embedding Layer
           ↓ Chaque mot → vecteur initial

        2. Attention Layer #1
           ↓ Les mots se "regardent" entre eux

        3. Feed Forward
           ↓ Transformation non-linéaire

        4. Attention Layer #2
           ↓ Encore plus de contexte

        ... (×12 couches) ...

        12. Attention Layer #12
           ↓ Compréhension profonde

        Output: Vecteurs contextuels riches!
        ```

        **Après 12 couches d'attention:**
        Le réseau a une compréhension **profonde** du sens de chaque mot dans son contexte! 🧠

        **Différence avec Word2Vec:**
        - Word2Vec: "banque" a **toujours** le même vecteur
        - BERT: "banque" a un vecteur **différent** selon le contexte! ✨
        """)

    with st.expander("📊 **Anatomie d'un Vecteur d'Embedding**"):
        st.markdown(f"""
        ### 🔬 Qu'y a-t-il dans un Vecteur?

        Un embedding de `{embedding_engine.embedding_dim}` dimensions, c'est quoi concrètement?
        Prenons un exemple réel!
        """)

        # Générer un embedding d'exemple
        sample_text = "Le chat noir mange du poisson"
        sample_embedding = embedding_engine.model.encode([sample_text])[0]

        col_graph, col_analysis = st.columns([3, 2])

        with col_graph:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Graphique 1: Distribution des valeurs
            ax1.hist(
                sample_embedding, bins=50, color="#3498db", alpha=0.7, edgecolor="black"
            )
            ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zéro")
            ax1.axvline(
                x=np.mean(sample_embedding),
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Moyenne ({np.mean(sample_embedding):.3f})",
            )
            ax1.set_xlabel("Valeur de la dimension", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Nombre de dimensions", fontsize=11, fontweight="bold")
            ax1.set_title(
                f"Distribution des {embedding_engine.embedding_dim} dimensions",
                fontsize=12,
                fontweight="bold",
            )
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Graphique 2: Échantillon des premières dimensions
            n_show = 50
            dims = np.arange(n_show)
            values = sample_embedding[:n_show]
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

            ax2.bar(
                dims, values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
            )
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
            ax2.set_xlabel("Index de la dimension", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Valeur", fontsize=11, fontweight="bold")
            ax2.set_title(
                f"Valeurs des {n_show} premières dimensions (vert=positif, rouge=négatif)",
                fontsize=11,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_analysis:
            st.markdown(f"""
            ### 📈 Analyse du Vecteur

            **Texte analysé:**
            > "{sample_text}"

            **Statistiques:**
            - **Dimensions:** {embedding_engine.embedding_dim}
            - **Valeur min:** {np.min(sample_embedding):.3f}
            - **Valeur max:** {np.max(sample_embedding):.3f}
            - **Moyenne:** {np.mean(sample_embedding):.3f}
            - **Écart-type:** {np.std(sample_embedding):.3f}

            **💡 Observations:**

            📊 **Graphique du haut:**
            - Distribution ~normale centrée autour de 0
            - Valeurs entre -1 et +1 (typique)
            - Pas de zéros → Dense! ✅

            📊 **Graphique du bas:**
            - Alternance positif/négatif
            - Chaque dimension = "concept"
            - Ex: Dim #12 = "animal"?
            - Ex: Dim #37 = "action"?

            **🧠 Chaque dimension capture:**
            - Syntaxe (nom, verbe, etc.)
            - Sémantique (animal, nourriture)
            - Relations (agent, patient)
            - Contexte culturel/linguistique

            C'est cette richesse qui permet la recherche sémantique! 🎯
            """)

    with st.expander("🔗 **Similarité Sémantique: Voir les Relations**"):
        st.markdown("""
        ### 🎯 Comment les Embeddings Capturent les Relations?

        Générons des embeddings pour plusieurs phrases et comparons-les!
        """)

        # Phrases d'exemple avec relations sémantiques
        example_phrases = [
            "Le chat mange du poisson",
            "Un chien dévore de la viande",
            "L'ordinateur calcule des nombres",
            "La voiture roule sur la route",
            "Le poisson nage dans l'eau",
            "Un félin chasse une souris",
        ]

        # Calculer les embeddings
        embeddings_matrix = embedding_engine.model.encode(example_phrases)

        # Calculer la matrice de similarité cosinus
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(embeddings_matrix)

        col_heatmap, col_explanation = st.columns([3, 2])

        with col_heatmap:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

            # Heatmap
            im = ax.imshow(
                similarity_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"
            )

            # Axes
            ax.set_xticks(np.arange(len(example_phrases)))
            ax.set_yticks(np.arange(len(example_phrases)))

            # Labels courts pour l'affichage
            short_labels = [
                "Chat/poisson",
                "Chien/viande",
                "Ordi/calcul",
                "Voiture/route",
                "Poisson/eau",
                "Félin/souris",
            ]

            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(short_labels, fontsize=9)

            # Annotations des valeurs
            for i in range(len(example_phrases)):
                for j in range(len(example_phrases)):
                    value = similarity_matrix[i, j]
                    color = "white" if value > 0.7 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.set_title(
                "Heatmap de Similarité Cosinus (Embeddings)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Similarité (0=différent, 1=identique)", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_explanation:
            st.markdown("""
            ### 🔍 Lecture de la Heatmap

            **Couleurs:**
            - 🟢 **Vert foncé:** Très similaire (~0.8-1.0)
            - 🟡 **Jaune:** Similaire (~0.5-0.8)
            - 🔴 **Rouge:** Peu similaire (~0.0-0.5)

            **💡 Observations Clés:**

            **Diagonale = 1.00 (vert):**
            - Chaque phrase comparée à elle-même
            - Similarité parfaite ✅

            **Relations sémantiques détectées:**
            """)

            # Trouver les paires les plus similaires (hors diagonale)
            similarity_no_diag = similarity_matrix.copy()
            np.fill_diagonal(similarity_no_diag, 0)

            # Top 3 paires
            flat_indices = np.argsort(similarity_no_diag.ravel())[::-1][:3]
            top_pairs = [
                (i // len(example_phrases), i % len(example_phrases))
                for i in flat_indices
            ]

            for rank, (i, j) in enumerate(top_pairs, 1):
                sim = similarity_matrix[i, j]
                st.success(f"""
                **#{rank} - Similarité: {sim:.3f}**
                - "{example_phrases[i][:30]}..."
                - "{example_phrases[j][:30]}..."
                """)

            st.markdown("""
            **🧠 Pourquoi ces relations?**

            Le modèle a appris que:
            - "chat" ≈ "chien" ≈ "félin" (animaux)
            - "mange" ≈ "dévore" ≈ "chasse" (actions)
            - "poisson" apparaît 2× (sujet et objet!)

            **⚠️ Notez bien:**
            - Phrases sans mots communs peuvent être similaires!
            - C'est la **sémantique**, pas le lexique!
            """)

    with st.expander("📚 **Comment le Réseau Apprend (Pré-entraînement)**"):
        st.markdown("""
        ### Masked Language Modeling (MLM)

        **Objectif:** Forcer le réseau à comprendre le contexte pour prédire des mots manquants.

        **Tâche d'entraînement:**
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1️⃣ Phrase originale**")
            st.code("Le chat mange du poisson", language="text")

        with col2:
            st.markdown("**2️⃣ Masquer un mot**")
            st.code("Le [MASK] mange du poisson", language="text")

        with col3:
            st.markdown("**3️⃣ Prédire**")
            st.code("Prédiction: chat", language="text")

        st.markdown("""
        ### 🤔 Pourquoi Ça Marche?

        Pour prédire correctement [MASK] = "chat", le réseau DOIT analyser:

        **Analyse syntaxique:**
        - "Le [MASK]" → Probablement un **nom** (article + nom)
        - "mange" → Le sujet doit être **vivant** (pas "table", "livre")

        **Analyse sémantique:**
        - "mange du poisson" → Animal qui mange du poisson
        - Options: chat, chien, ours, humain
        - Dans ce contexte: **chat** est le plus probable! 🎯

        **Analyse contextuelle:**
        - Langue: français (pas "cat", "gato")
        - Registre: langue courante (pas jargon technique)

        ---

        ### 📊 Exemples d'Entraînement Réels
        """)

        training_examples = pd.DataFrame(
            {
                "Phrase Maskée": [
                    "Paris est la [MASK] de la France",
                    "Einstein a découvert la théorie de la [MASK]",
                    "Le [MASK] est un fruit rouge",
                    "J'aime coder en [MASK] pour le web",
                ],
                "Prédiction": [
                    "capitale",
                    "relativité",
                    "fraise / tomate",
                    "JavaScript / Python",
                ],
                "Difficulté": [
                    "⭐ Facile",
                    "⭐⭐ Moyen",
                    "⭐⭐ Moyen",
                    "⭐⭐⭐ Difficile",
                ],
            }
        )

        st.dataframe(training_examples, use_container_width=True, hide_index=True)

        st.markdown("""
        ### 🎓 Ce Que le Réseau Apprend

        **Après des milliards d'exemples:**

        1. **Syntaxe:** Structure des phrases (sujet-verbe-complément)
        2. **Sémantique:** Relations entre concepts (capitale ↔ pays)
        3. **Connaissances factuelles:** Paris est la capitale de France
        4. **Contexte:** Mots qui vont ensemble (coder → JavaScript/Python)

        **Résultat:**
        Le réseau apprend des **représentations vectorielles riches** qui capturent le sens! ✨

        **Comparaison avec TF-IDF:**
        - TF-IDF: Compte les mots (aucun apprentissage)
        - BERT: Apprend le sens via des milliards d'exemples! 🔥
        """)

    with st.expander("🌈 **Les Dimensions: Que Représentent-elles?**"):
        st.markdown("""
        ### L'Espace Vectoriel à 384 Dimensions

        **Question fondamentale:** Qu'est-ce que ces 384 nombres représentent? 🤔

        **Réponse courte:** Des **concepts sémantiques** appris automatiquement!

        ---

        ### 🎨 Exemple Simplifié (Illustration)

        **Note:** Les vraies dimensions sont beaucoup plus complexes, mais voici l'intuition:
        """)

        dim_examples = pd.DataFrame(
            {
                "Dimension": [
                    "Dim 0",
                    "Dim 1",
                    "Dim 2",
                    "Dim 3",
                    "Dim 4",
                    "...",
                    "Dim 383",
                ],
                "Concept (Simplifié)": [
                    "vivant ↔ non-vivant",
                    "concret ↔ abstrait",
                    "positif ↔ négatif",
                    "animal ↔ objet",
                    "action ↔ état",
                    "...",
                    "??? (complexe)",
                ],
                "Exemple +": [
                    "chat (+0.9)",
                    "pomme (+0.8)",
                    "heureux (+0.9)",
                    "chien (+0.85)",
                    "courir (+0.7)",
                    "...",
                    "???",
                ],
                "Exemple −": [
                    "table (−0.7)",
                    "amour (−0.6)",
                    "triste (−0.8)",
                    "voiture (−0.75)",
                    "dormir (−0.6)",
                    "...",
                    "???",
                ],
            }
        )

        st.dataframe(dim_examples, use_container_width=True, hide_index=True)

        st.markdown("""
        ### ⚠️ Attention: Simplification!

        En réalité, **aucune dimension n'est aussi simple**.

        **Chaque dimension** capture une **combinaison complexe** de milliers de concepts:
        - Syntaxe + sémantique + contexte
        - Relations multiples simultanées
        - Interactions non-linéaires

        **Exemple réel:**
        - Dimension 42 pourrait capturer: "animal domestique + affection + relation humaine"
        - Pas juste "animal" ou "domestique" séparément

        ---

        ### 🔬 Comment les Dimensions Émergent

        **Le réseau n'est PAS programmé avec ces concepts!**

        **Processus d'apprentissage:**

        1. **Initialisation:** Valeurs aléatoires
        2. **Entraînement:** Millions d'exemples de texte
        3. **Ajustement:** Le réseau ajuste les poids pour mieux prédire
        4. **Émergence:** Les dimensions se spécialisent naturellement!

        **Exemple concret:**
        ```
        Le réseau voit:
        - "chat" apparaît avec "miaule", "ronronne", "souris"
        - "chien" apparaît avec "aboie", "queue", "maître"
        - "table" apparaît avec "bois", "chaise", "manger"

        Après entraînement:
        - Dimension X encode "animalité" (chat et chien proche, table loin)
        - Dimension Y encode "domestique" (chat, chien, et table proche!)
        - Dimension Z encode "mobilité" (chat et chien proche, table loin)

        Résultat: "chat" et "chien" sont proches dans l'espace!
        ```

        ---

        ### 🎯 Ce Qui Importe

        **Peu importe ce que chaque dimension représente individuellement!**

        **Ce qui compte:**
        - Les **relations géométriques** entre vecteurs
        - "chat" et "chien" sont **proches** (petite distance)
        - "chat" et "ordinateur" sont **éloignés** (grande distance)

        **Magie des embeddings:** Les relations sémantiques émergent naturellement! 🪄

        **Analogie:**
        - Tu n'as pas besoin de comprendre comment fonctionne chaque neurone de ton cerveau
        - Ce qui compte c'est que tu puisses reconnaître un chat! 🐱
        """)

    with st.expander("⚔️ **Battle: TF-IDF vs Embeddings**"):
        st.markdown("""
        ### 🥊 Le Test Ultime: Comprendre la Différence

        Prenons des **paires de phrases** et comparons les similarités selon:
        - **Approche Lexicale** (TF-IDF) → Compte les mots communs
        - **Approche Sémantique** (Embeddings) → Comprend le sens
        """)

        # Paires de test
        test_pairs = [
            ("Un chat noir dort", "Le félin sombre se repose", "Synonymes parfaits"),
            (
                "Je cuisine un plat italien",
                "Je prépare une recette de pâtes",
                "Même sujet",
            ),
            (
                "Paris est belle",
                "La capitale française est magnifique",
                "Référence identique",
            ),
            ("Le chien aboie fort", "La table est en bois", "Aucun rapport"),
            ("J'adore la programmation", "Je déteste coder", "Contraires"),
            ("Voiture rapide rouge", "Automobile véloce écarlate", "Synonymes exacts"),
        ]

        # Calculer les similarités
        results_data = []

        for phrase1, phrase2, description in test_pairs:
            # Embedding similarity
            emb1, emb2 = embedding_engine.model.encode([phrase1, phrase2])
            emb_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # TF-IDF similarity (approximation simple avec Jaccard sur les mots)
            words1 = set(phrase1.lower().split())
            words2 = set(phrase2.lower().split())
            if len(words1.union(words2)) > 0:
                jaccard_sim = len(words1.intersection(words2)) / len(
                    words1.union(words2)
                )
            else:
                jaccard_sim = 0.0

            results_data.append(
                {
                    "Description": description,
                    "TF-IDF (lexical)": jaccard_sim,
                    "Embeddings (sémantique)": emb_sim,
                    "Différence": abs(emb_sim - jaccard_sim),
                }
            )

        # Graphique comparatif
        col_graph, col_analysis = st.columns([3, 2])

        with col_graph:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 7))

            x_pos = np.arange(len(results_data))
            width = 0.35

            tfidf_scores = [d["TF-IDF (lexical)"] for d in results_data]
            emb_scores = [d["Embeddings (sémantique)"] for d in results_data]
            labels = [d["Description"] for d in results_data]

            # Barres
            ax.barh(
                x_pos - width / 2,
                tfidf_scores,
                width,
                label="TF-IDF (lexical)",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
            )
            ax.barh(
                x_pos + width / 2,
                emb_scores,
                width,
                label="Embeddings (sémantique)",
                color="#2ecc71",
                alpha=0.8,
                edgecolor="black",
            )

            # Annotations
            for i, (tf, emb) in enumerate(zip(tfidf_scores, emb_scores)):
                ax.text(
                    tf + 0.02,
                    i - width / 2,
                    f"{tf:.2f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
                ax.text(
                    emb + 0.02,
                    i + width / 2,
                    f"{emb:.2f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_yticks(x_pos)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel(
                "Score de Similarité (0=différent, 1=identique)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_title(
                "Comparaison: TF-IDF vs Embeddings",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax.legend(fontsize=10, loc="lower right")
            ax.grid(True, alpha=0.3, axis="x")
            ax.set_xlim(0, 1.1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_analysis:
            st.markdown("""
            ### 🔍 Analyse des Résultats

            **Paires critiques:**
            """)

            # Trouver les cas où embeddings >> TF-IDF
            for data in results_data:
                if data["Embeddings (sémantique)"] > data["TF-IDF (lexical)"] + 0.2:
                    st.success(f"""
                    **{data["Description"]}**
                    - TF-IDF: {data["TF-IDF (lexical)"]:.2f}
                    - Embeddings: {data["Embeddings (sémantique)"]:.2f}
                    - ✅ Embeddings gagne!
                    """)

            st.markdown("""
            ---

            **💡 Ce que ça montre:**

            **Cas "Synonymes parfaits":**
            - Phrases signifient la MÊME chose
            - TF-IDF: ~0.0 (aucun mot commun!)
            - Embeddings: ~0.8 (sens identique!) ✨

            **Cas "Contraires":**
            - "adore" vs "déteste" → opposés
            - TF-IDF: Pense que c'est similaire
            - Embeddings: Détecte l'opposition! 🎯

            **Cas "Aucun rapport":**
            - Les deux méthodes concordent
            - Peu de mots communs = peu de sens commun

            ---

            **🏆 Verdict:**

            **TF-IDF:**
            - Bon pour correspondance exacte
            - Rapide et simple
            - Limité au lexique

            **Embeddings:**
            - Comprend les synonymes ✅
            - Capture le sens profond ✅
            - Détecte les nuances ✅
            - **Mais:** Plus lent et complexe
            """)

        st.divider()

        st.info("""
        **🎓 Conclusion Pédagogique**

        Les embeddings ne sont PAS magiques! Ils ont simplement appris à:
        1. Reconnaître que "chat" et "félin" sont liés (via des milliards d'exemples)
        2. Placer ces mots proches dans l'espace vectoriel
        3. Étendre cette logique à des phrases entières

        **Résultat:** Recherche sémantique = comprendre l'intention, pas juste les mots! 🚀
        """)


def render_embeddings_search(
    embedding_engine, documents_texts, documents_titles, documents_categories
):
    """Recherche interactive avec embeddings"""
    st.header("🔍 Recherche Sémantique Interactive")

    st.markdown("""
    Teste la puissance de la recherche sémantique!
    Essaie des **synonymes**, des **paraphrases**, des **concepts**! 🚀
    """)

    # Utiliser un formulaire pour éviter les problèmes de rerun
    with st.form("emb_search_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "🔎 Ta recherche:",
                value="animal domestique fidèle",  # Valeur par défaut!
                placeholder="animal domestique, cuisine italienne, technologie moderne...",
                key="emb_query_input",
                help='💡 **Exemples:** "animal domestique fidèle" | "cuisine italienne traditionnelle" | "technologie moderne innovation" | "voyage aventure exotique"',
            )

        with col2:
            top_k = st.slider("Résultats:", 3, 20, 5, key="emb_topk_slider")

        # Bouton de soumission (Enter fonctionne aussi!)
        submitted = st.form_submit_button(
            "🚀 Rechercher Sémantiquement!", type="primary"
        )

    if submitted and query:
        with st.spinner("🧠 Recherche sémantique en cours..."):
            results = embedding_engine.search(query, top_k=top_k)

            if len(results) == 0:
                st.warning("😕 Aucun résultat trouvé!")
            else:
                st.success(f"✅ {len(results)} résultats trouvés!")

                # Graphique des scores
                doc_indices = [r["index"] for r in results]
                scores = [r["score"] for r in results]
                labels = [
                    documents_titles[idx][:40] + "..."
                    if len(documents_titles[idx]) > 40
                    else documents_titles[idx]
                    for idx in doc_indices
                ]

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 6))
                y_pos = np.arange(len(labels))
                bars = ax.barh(y_pos, scores, color="#1f77b4", edgecolor="black")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10)
                ax.invert_yaxis()
                ax.set_xlabel("Score de Similarité Sémantique", fontweight="bold")
                ax.set_title(
                    f'Top {len(results)} Résultats pour: "{query}"',
                    fontsize=13,
                    fontweight="bold",
                )
                ax.grid(axis="x", alpha=0.3)

                for i, (bar, score) in enumerate(zip(bars, scores)):
                    width = bar.get_width()
                    ax.text(
                        width,
                        i,
                        f" {score:.3f}",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

                plt.tight_layout()
                st.pyplot(fig)

                # Détails des résultats avec analyses enrichies
                st.markdown("### 🎯 Résultats Détaillés")

                # Analyse de distribution des scores
                scores_list = [r["score"] for r in results]
                avg_score = np.mean(scores_list)
                max_score = scores_list[0]
                min_score = scores_list[-1]
                score_range = max_score - min_score

                st.markdown(f"""
                **📊 Analyse rapide des scores:**
                - **Meilleur:** {max_score:.3f} {"🔥" if max_score > 0.7 else "✅" if max_score > 0.5 else "⚠️"}
                - **Moyen:** {avg_score:.3f}
                - **Pire:** {min_score:.3f}
                - **Écart:** {score_range:.3f} {"(bonne séparation!)" if score_range > 0.2 else "(scores proches)"}
                """)

                for rank, result in enumerate(results, 1):
                    doc_idx = result["index"]
                    score = result["score"]

                    # Badge selon le score
                    if score > 0.7:
                        badge = "🔥"
                        quality = "Excellent"
                    elif score > 0.5:
                        badge = "✅"
                        quality = "Très bon"
                    elif score > 0.3:
                        badge = "👌"
                        quality = "Bon"
                    else:
                        badge = "⚠️"
                        quality = "Faible"

                    with st.expander(
                        f"{badge} **#{rank}** - {documents_titles[doc_idx]} • Similarité: **{score:.3f}** ({quality})"
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.caption(f"📁 Catégorie: {documents_categories[doc_idx]}")
                            st.write(documents_texts[doc_idx][:400] + "...")

                        with col2:
                            st.markdown("**📊 Analyse:**")
                            st.metric("Score", f"{score:.3f}", f"{score * 100:.1f}%")

                            # Position relative
                            position_pct = (
                                (score - min_score) / score_range
                                if score_range > 0
                                else 0
                            )
                            st.metric("Position", f"Top {position_pct * 100:.0f}%")

                            # Comparaison avec la moyenne
                            diff_avg = score - avg_score
                            st.metric("vs Moyenne", f"{diff_avg:+.3f}")

                        st.markdown("---")

                        st.info(f"""
                        **💡 Interprétation du score {score:.3f}:**

                        **Similarité sémantique:** {score * 100:.1f}%

                        **Ce que ça signifie:**
                        - {"> 0.7: Documents **très similaires**! Même sujet, vocabulaire proche 🔥" if score > 0.7 else ""}
                        - {"0.5-0.7: Documents **similaires**. Sujets connexes, concepts liés ✅" if 0.5 < score <= 0.7 else ""}
                        - {"0.3-0.5: Documents **moyennement similaires**. Quelques concepts communs 👌" if 0.3 < score <= 0.5 else ""}
                        - {"< 0.3: Documents **peu similaires**. Sujets différents ⚠️" if score <= 0.3 else ""}

                        **Pourquoi ce rang?**
                        - Embeddings capture le **sens global** du texte
                        - Pas besoin de mots identiques (synonymes OK!)
                        - Relations sémantiques implicites détectées 🎯
                        """)

                # Conseils pédagogiques
                st.markdown("---")
                st.success("""
                **💡 Expérimente avec différentes queries!**

                **Astuce 1:** Teste des **synonymes**
                - Query: "voiture rapide" vs "automobile véloce"
                - Embeddings devrait donner des résultats similaires! ✅

                **Astuce 2:** Teste des **concepts**
                - Query: "capitale France" → Devrait trouver "Paris"!
                - TF-IDF ne peut PAS faire ça (aucun mot commun)

                **Astuce 3:** Teste des **paraphrases**
                - "recette italienne pâtes" vs "plat italien spaghetti"
                - Sens identique, mots différents → Embeddings comprend! 🔥
                """)


def render_embeddings_exploration(
    embedding_engine, documents_texts, documents_titles, documents_categories
):
    """Exploration et visualisations de l'espace vectoriel"""
    st.header("📊 Exploration de l'Espace Vectoriel")

    st.markdown("### 🌌 Visualisation 3D Interactive (PCA)")

    # Paramètres visualisation
    col1, col2 = st.columns([2, 1])

    with col1:
        viz_query = st.text_input(
            "Query à visualiser (optionnel):",
            placeholder="cuisine italienne",
            key="viz_query",
            help="💡 Si fournie, la query sera affichée sur le graphique 3D",
        )

    with col2:
        n_docs_viz = st.slider(
            "Nombre de docs:", 10, 100, min(30, len(documents_texts)), key="n_docs_viz"
        )

    if st.button("🎨 Générer la visualisation 3D!", key="viz_3d_btn"):
        with st.spinner("🎨 Génération de la visualisation 3D..."):
            embeddings_subset = embedding_engine.get_embeddings()[:n_docs_viz]
            labels_subset = documents_titles[:n_docs_viz]
            categories_subset = (
                documents_categories[:n_docs_viz] if documents_categories else None
            )

            if viz_query:
                query_emb = embedding_engine.get_query_embedding(viz_query)
                results = embedding_engine.search(viz_query, top_k=5)
                top_indices = [r["index"] for r in results if r["index"] < n_docs_viz]
            else:
                query_emb = None
                top_indices = None

            fig_3d = plot_embedding_space_3d(
                embeddings_subset,
                labels_subset,
                categories=categories_subset,
                query_embedding=query_emb,
                query_label=viz_query if viz_query else "Query",
                top_k_indices=top_indices,
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            st.info("""
            💡 **Interprétation:**
            - Chaque point = un document
            - Documents similaires sont **proches** dans l'espace
            - Les couleurs représentent les catégories
            - La query (si fournie) est en rouge 🔴
            - Les lignes vertes montrent les top résultats
            """)

    st.divider()

    # Clustering automatique
    st.markdown("### 🎯 Clustering Automatique des Documents")

    n_clusters = st.slider("Nombre de clusters:", 2, 10, 3, key="n_clusters")

    if st.button("🧩 Calculer les clusters!", key="cluster_btn"):
        with st.spinner("🧩 Clustering en cours..."):
            embeddings_all = embedding_engine.get_embeddings()

            fig_cluster = plot_clustering_2d(
                embeddings_all, documents_titles, n_clusters=n_clusters
            )
            st.pyplot(fig_cluster)

            # Afficher quelques docs par cluster
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings_all)

            st.markdown("### 📑 Documents par Cluster")

            for cluster_id in range(n_clusters):
                with st.expander(
                    f"🎨 Cluster {cluster_id + 1} ({sum(clusters == cluster_id)} documents)"
                ):
                    cluster_docs = [
                        i for i, c in enumerate(clusters) if c == cluster_id
                    ]
                    for doc_id in cluster_docs[:5]:  # Top 5
                        st.write(f"- **{documents_titles[doc_id]}**")
                        st.caption(f"  {documents_texts[doc_id][:100]}...")

    st.divider()

    # Documents similaires
    st.markdown("### 🔗 Explorer les Similarités")

    selected_doc_idx = st.selectbox(
        "Choisir un document:",
        options=range(min(50, len(documents_titles))),
        format_func=lambda i: f"{documents_titles[i][:60]}...",
        key="sim_doc_select",
    )

    if st.button("🔍 Trouver documents similaires!", key="find_sim_btn"):
        with st.spinner("🔍 Recherche de documents similaires..."):
            similar_docs = embedding_engine.find_similar(selected_doc_idx, top_k=5)

            st.markdown("**📄 Document source:**")
            st.info(
                f"**{documents_titles[selected_doc_idx]}**\n\n{documents_texts[selected_doc_idx][:200]}..."
            )

            st.markdown("**Documents similaires:**")

            for i, sim_doc in enumerate(similar_docs, 1):
                idx = sim_doc["index"]
                score = sim_doc["score"]

                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.metric(f"#{i}", f"{score:.3f}")
                    with col2:
                        st.markdown(f"**{documents_titles[idx]}**")
                        st.caption(documents_texts[idx][:150] + "...")


def render_embeddings_stepbystep(embedding_engine, documents_texts, documents_titles):
    """Exemple pas-à-pas complet avec PÉDAGOGIE MAXIMALE"""
    st.header("🎓 Exemple Complet: De A à Z")

    st.markdown("""
    Dans cette section, on va dérouler **TOUT** le processus des embeddings sur un exemple simple.

    Tu vas voir:
    1. Comment le texte devient un vecteur (encoding)
    2. Les calculs mathématiques exacts
    3. Comment on mesure la similarité
    4. Pourquoi ça marche si bien! 🔍

    **Objectif:** Comprendre chaque étape du pipeline embeddings! 🎯
    """)

    # Mini corpus
    corpus_example = documents_texts[:3]
    query_example = st.text_input(
        "🔎 Ta query:",
        value="cuisine traditionnelle",
        key="tutorial_query",
        help="💡 Teste avec différentes queries pour voir comment les calculs changent",
    )

    st.markdown("### 📝 Setup")
    st.code(f"""
Corpus ({len(corpus_example)} documents):
{chr(10).join(f'  Doc {i}: "{doc[:80]}..."' for i, doc in enumerate(corpus_example))}

Query: "{query_example}"
    """)

    if query_example:
        # Étape 1: Calcul embeddings
        st.markdown("### 1️⃣ Calcul des Embeddings")

        with st.spinner("🧠 Calcul des vecteurs..."):
            query_emb = embedding_engine.get_query_embedding(query_example)
            doc_embs = embedding_engine.get_embeddings()[:3]

        st.success(f"✅ Embeddings calculés! Dimensions: {len(query_emb)}")

        # Afficher quelques valeurs
        for i, doc in enumerate(corpus_example):
            with st.expander(f"📄 Doc {i}: {documents_titles[i]}"):
                vec = doc_embs[i]
                st.code(f"""
Vector ({len(vec)} dimensions):
[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, ..., {vec[-1]:.3f}]

Premiers 10 éléments:
{vec[:10]}
                """)

        with st.expander(f"🔎 Query: {query_example}"):
            st.code(f"""
Vector ({len(query_emb)} dimensions):
[{query_emb[0]:.3f}, {query_emb[1]:.3f}, {query_emb[2]:.3f}, ..., {query_emb[-1]:.3f}]

Premiers 10 éléments:
{query_emb[:10]}
            """)

        # Étape 2: Calcul similarités
        st.markdown("### 2️⃣ Calcul de Similarité Cosinus")

        st.latex(r"\text{sim}(q, d) = \frac{q \cdot d}{||q|| \times ||d||}")

        # Calcul détaillé pour Doc 0
        st.markdown("**Exemple détaillé pour Doc 0:**")

        with st.expander("📐 Calculs étape par étape"):
            dot_product = np.dot(query_emb, doc_embs[0])
            norm_q = np.linalg.norm(query_emb)
            norm_d = np.linalg.norm(doc_embs[0])
            similarity = dot_product / (norm_q * norm_d)

            st.code(f"""
1. Produit scalaire (dot product):
   q · d = {dot_product:.6f}

2. Norme de la query:
   ||q|| = √({norm_q**2:.6f}) = {norm_q:.6f}

3. Norme du document:
   ||d|| = √({norm_d**2:.6f}) = {norm_d:.6f}

4. Similarité cosinus:
   sim = {dot_product:.6f} / ({norm_q:.6f} × {norm_d:.6f})
       = {dot_product:.6f} / {norm_q * norm_d:.6f}
       = {similarity:.6f}
            """)

        # Étape 3: Résultats
        st.markdown("### 3️⃣ Résultats Finaux")

        results = embedding_engine.search(query_example, top_k=3)

        results_data = []
        for rank, result in enumerate(results, 1):
            results_data.append(
                {
                    "Rang": rank,
                    "Document": documents_titles[result["index"]][:50],
                    "Score": f"{result['score']:.4f}",
                }
            )

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        st.success(f"""
        ✅ **Résultat:**

        Le document "{results_data[0]["Document"]}" est le plus similaire!

        **Pourquoi?**
        - Embeddings capture le **sens** et non les mots exacts
        - Comprend les concepts, synonymes, et relations sémantiques! 🎯
        """)


def render_embeddings_comparison(
    embedding_engine, tfidf_engine, bm25_engine, documents_texts, documents_titles
):
    """Comparaison Embeddings vs TF-IDF vs BM25"""
    st.header("⚔️ Battle Royale: Embeddings vs BM25 vs TF-IDF")

    st.markdown("""
    Compare les 3 techniques sur une même requête!

    💡 **Astuce:** Essaie des queries avec synonymes ou concepts pour voir la différence!
    """)

    # Utiliser un formulaire pour éviter les problèmes de rerun
    with st.form("battle_form", clear_on_submit=False):
        battle_query = st.text_input(
            "🔎 Query de comparaison:",
            value="nourriture italienne pâtes",
            key="battle_query_input",
            help='💡 **Exemples:** "nourriture italienne pâtes" | "science-fiction futur" | "sport football compétition"',
        )

        top_k_battle = st.slider(
            "Nombre de résultats:", 5, 20, 10, key="battle_topk_slider"
        )

        # Bouton de soumission (Enter fonctionne aussi!)
        battle_submitted = st.form_submit_button(
            "⚔️ LANCER LA BATAILLE!", type="primary"
        )

    if battle_submitted and battle_query:
        with st.spinner("⚔️ Comparaison en cours..."):
            # Lancer les 3 techniques
            results_tfidf = tfidf_engine.search(battle_query, top_k=top_k_battle)
            results_bm25 = bm25_engine.search(battle_query, top_k=top_k_battle)
            results_embeddings_raw = embedding_engine.search(
                battle_query, top_k=top_k_battle
            )

            # Convertir embeddings au format (idx, score)
            results_embeddings = [
                (r["index"], r["score"]) for r in results_embeddings_raw
            ]

            # Visualisation comparative
            results_dict = {
                "TF-IDF": results_tfidf,
                "BM25": results_bm25,
                "Embeddings": results_embeddings,
            }

            fig_comp = plot_multi_technique_comparison(
                results_dict, documents_titles, battle_query, top_k=top_k_battle
            )
            st.pyplot(fig_comp)

            # Métriques de comparaison
            st.divider()
            st.markdown("### 📈 Métriques de Comparaison")

            set_tfidf = set([idx for idx, _ in results_tfidf])
            set_bm25 = set([idx for idx, _ in results_bm25])
            set_emb = set([idx for idx, _ in results_embeddings])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                overlap_tb = len(set_tfidf & set_bm25)
                st.metric("TF-IDF ∩ BM25", f"{overlap_tb}/{top_k_battle}")

            with col2:
                overlap_te = len(set_tfidf & set_emb)
                st.metric("TF-IDF ∩ Embeddings", f"{overlap_te}/{top_k_battle}")

            with col3:
                overlap_be = len(set_bm25 & set_emb)
                st.metric("BM25 ∩ Embeddings", f"{overlap_be}/{top_k_battle}")

            with col4:
                overlap_all = len(set_tfidf & set_bm25 & set_emb)
                st.metric("Commun aux 3", f"{overlap_all}/{top_k_battle}")

            st.info("""
            💡 **Interprétation:**

            - **Overlap faible** entre Embeddings et TF-IDF/BM25 → Embeddings trouve des résultats **différents** (sémantiques)
            - **Overlap élevé** → Les 3 techniques s'accordent sur les meilleurs résultats
            - **Documents uniques à Embeddings** → Probablement trouvés par **synonymes ou concepts**!
            """)


def render_embeddings_hybrid(
    embedding_engine,
    bm25_engine,
    documents_texts,
    documents_titles,
    documents_categories,
):
    """Hybrid Search: BM25 + Embeddings"""
    st.header("🎨 Hybrid Search: Le Meilleur des Deux Mondes")

    st.markdown("""
    ### 🤝 Combiner Lexical (BM25) et Sémantique (Embeddings)

    **Principe:**
    ```python
    score_final = α × score_bm25 + (1-α) × score_embeddings
    ```

    Où **α** contrôle le poids de chaque technique!
    """)

    # Créer hybrid engine
    hybrid_engine = create_hybrid_engine(
        documents_texts, bm25_engine, embedding_engine, alpha=0.5
    )

    # Widget de tuning
    alpha = st.slider(
        "α (poids BM25):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = Embeddings pur | 1 = BM25 pur | 0.5 = équilibré",
        key="hybrid_alpha",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Poids BM25 (lexical)", f"{alpha:.0%}")
    with col2:
        st.metric("Poids Embeddings (sémantique)", f"{(1 - alpha):.0%}")

    # Utiliser un formulaire pour éviter les problèmes de rerun
    with st.form("hybrid_form", clear_on_submit=False):
        hybrid_query = st.text_input(
            "🔎 Recherche hybrid:",
            value="cuisine traditionnelle maison",  # Valeur par défaut!
            placeholder="smartphone dernière génération",
            key="hybrid_query_input",
            help='💡 **Exemples:** "smartphone dernière génération" | "cuisine traditionnelle maison" | "voiture électrique performante"',
        )

        top_k_hybrid = st.slider("Résultats:", 5, 20, 10, key="hybrid_topk_slider")

        # Bouton de soumission (Enter fonctionne aussi!)
        hybrid_submitted = st.form_submit_button(
            "🚀 Rechercher Hybrid!", type="primary"
        )

    if hybrid_submitted and hybrid_query:
        with st.spinner("🎨 Recherche hybrid en cours..."):
            results_hybrid = hybrid_engine.search(
                hybrid_query, top_k=top_k_hybrid, alpha=alpha
            )

            st.success(f"✅ {len(results_hybrid)} résultats trouvés!")

            # Affichage des résultats
            st.markdown("### 🏆 Résultats Hybrid")

            for i, result in enumerate(results_hybrid, 1):
                doc_idx = result["index"]

                with st.expander(
                    f"#{i} - {documents_titles[doc_idx]} (Score: {result['combined_score']:.3f})"
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Score BM25", f"{result['bm25_score']:.3f}")
                    with col2:
                        st.metric("Score Embeddings", f"{result['emb_score']:.3f}")
                    with col3:
                        st.metric("Score Combiné", f"{result['combined_score']:.3f}")

                    st.caption(f"📁 {documents_categories[doc_idx]}")
                    st.write(documents_texts[doc_idx][:250] + "...")

            # Visualisation de l'effet de alpha
            st.divider()
            st.markdown("### 📊 Impact du Paramètre α")

            # Calculer scores pour différents alpha
            alpha_values = np.linspace(0, 1, 21)
            sample_doc_idx = results_hybrid[0]["index"]  # Premier résultat
            doc_scores = []

            for a in alpha_values:
                score = hybrid_engine.compute_score(
                    hybrid_query, sample_doc_idx, alpha=a
                )
                doc_scores.append(score)

            fig_alpha = plot_hybrid_alpha_effect(
                alpha_values, doc_scores, alpha, documents_titles[sample_doc_idx][:50]
            )
            st.pyplot(fig_alpha)

            st.info("""
            💡 **Quand ajuster α?**

            - **α ≈ 0.3-0.4:** Corpus avec beaucoup de synonymes, recherche conceptuelle
            - **α ≈ 0.5-0.6:** Équilibré (recommandé par défaut) ⭐
            - **α ≈ 0.7-0.8:** Noms exacts importants, codes, IDs
            """)


def render_embeddings_performance(
    embedding_engine, documents_texts, tfidf_engine, bm25_engine
):
    """Performance et optimisations des embeddings - VERSION PÉDAGOGIQUE"""
    st.header("⚡ Analyse des Performances")

    st.info("""
    **💡 Disclaimer Important:**

    Les embeddings sont **plus lents** que TF-IDF/BM25, MAIS offrent des résultats **beaucoup meilleurs**!

    **Trade-off:** Vitesse vs Qualité
    - TF-IDF/BM25: Rapide mais limité (lexical matching)
    - Embeddings: Plus lent mais puissant (semantic matching) 🎯
    """)

    st.markdown("---")
    st.markdown("### ⏱️ Comparaison des Temps de Calcul")

    # Tableau comparatif enrichi
    perf_data = {
        "Opération": [
            "Indexation (1000 docs)",
            "Recherche (1 query)",
            "Recherche (100 queries)",
            "Mémoire (1000 docs)",
            "Scalabilité (10k docs)",
        ],
        "TF-IDF": ["~0.1s ⚡", "~5ms ⚡", "~0.5s ⚡", "~2 MB", "Linéaire"],
        "BM25": ["~0.1s ⚡", "~5ms ⚡", "~0.5s ⚡", "~2 MB", "Linéaire"],
        "Embeddings (CPU)": ["~300s 🐌", "~10ms", "~1s", "~15 MB", "GPU requis!"],
        "Embeddings (GPU)": ["~30s ⚡⚡", "~10ms", "~1s", "~15 MB", "OK jusqu'à 100k"],
    }

    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("""
    ### 📊 Observations Clés

    **Indexation (Calcul des Embeddings):**
    - **TF-IDF/BM25:** Instantané (~0.1s pour 1000 docs) ⚡
    - **Embeddings (CPU):** TRÈS lent (~300s pour 1000 docs) 🐌
    - **Embeddings (GPU):** Acceptable (~30s pour 1000 docs) ⚡⚡

    **⚠️ Pourquoi si lent?**
    - Chaque document passe par un réseau de neurones (BERT)
    - 12 couches d'attention + millions de paramètres
    - Calculs intensifs (multiplications matricielles)

    **Recherche (Query → Résultats):**
    - Toutes les méthodes sont rapides (<10ms)
    - Embeddings: juste un calcul de distance (produit scalaire)
    - Une fois indexé, la recherche est instantanée! ✅

    **Mémoire:**
    - TF-IDF/BM25: Matrice sparse (beaucoup de zéros)
    - Embeddings: Matrice dense (pas de zéros, plus gros)
    - Trade-off: 5-10× plus de RAM pour embeddings
    """)

    st.warning("""
    ⚠️ **Verdict: Quand utiliser Embeddings?**

    **OUI si:**
    - Tu as un GPU ou patience (indexation lente acceptable)
    - Tu veux la MEILLEURE qualité de recherche
    - Ton corpus contient synonymes/paraphrases/concepts
    - Tu indexes une fois, recherches souvent

    **NON si:**
    - Tu n'as pas de GPU ET corpus énorme (>10k docs)
    - Tu réindexes fréquemment (données changeantes)
    - TF-IDF/BM25 suffit déjà (mots-clés simples)
    - Contraintes temps réel strictes

    **Compromis Hybrid:** Utilise les deux! (voir onglet "Hybrid") 🎯
    """)

    st.divider()

    # Benchmark réel
    st.markdown("### 🏁 Benchmark Réel")

    if st.button("🚀 Lancer un benchmark!", key="benchmark_btn"):
        with st.spinner("⏱️ Benchmarking en cours..."):
            n_test_queries = 5
            test_queries = ["cuisine", "technologie", "histoire", "sport", "culture"][
                :n_test_queries
            ]

            times_search = {"TF-IDF": [], "BM25": [], "Embeddings": []}

            for query in test_queries:
                # TF-IDF
                start = time.time()
                tfidf_engine.search(query, top_k=10)
                times_search["TF-IDF"].append(time.time() - start)

                # BM25
                start = time.time()
                bm25_engine.search(query, top_k=10)
                times_search["BM25"].append(time.time() - start)

                # Embeddings
                start = time.time()
                embedding_engine.search(query, top_k=10)
                times_search["Embeddings"].append(time.time() - start)

            # Résultats
            st.markdown("### 📊 Résultats")

            avg_times = {k: np.mean(v) * 1000 for k, v in times_search.items()}  # en ms

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            techniques = list(avg_times.keys())
            times = list(avg_times.values())
            colors = ["#d62728", "#2ca02c", "#1f77b4"]
            bars = ax.bar(
                techniques, times, color=colors, edgecolor="black", linewidth=1.5
            )

            ax.set_ylabel("Temps Moyen (millisecondes)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"Temps de Recherche (moyenne sur {n_test_queries} queries)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.3)

            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time_val:.1f}ms",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            plt.tight_layout()
            st.pyplot(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("TF-IDF", f"{avg_times['TF-IDF']:.2f}ms")
            with col2:
                st.metric("BM25", f"{avg_times['BM25']:.2f}ms")
            with col3:
                st.metric(
                    "Embeddings",
                    f"{avg_times['Embeddings']:.2f}ms",
                    delta=f"+{avg_times['Embeddings'] - avg_times['BM25']:.1f}ms",
                    delta_color="inverse",
                )

    st.divider()

    # Optimisations
    st.markdown("### 🚀 Optimisations Possibles")

    st.markdown("""
    #### 1. **Utiliser un GPU** ⚡
    ```python
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('model-name', device=device)
    ```
    **Speedup:** 10-50× plus rapide!

    #### 2. **Batch Processing**
    ```python
    embeddings = model.encode(documents, batch_size=32)
    ```
    **Speedup:** 2-5× plus rapide

    #### 3. **Utiliser FAISS (Vector Database)**
    ```python
    import faiss
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    scores, indices = index.search(query_embedding, k=10)
    ```
    **Speedup:** 10-100× sur gros corpus (millions de docs)

    #### 4. **Modèles Plus Petits**

    | Modèle | Dimensions | Vitesse | Qualité |
    |--------|-----------|---------|---------|
    | MiniLM | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
    | MPNet | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
    | Large | 1024 | ⚡ | ⭐⭐⭐⭐⭐ |

    **Recommandation:** MiniLM pour la plupart des cas! 🎯

    #### 5. **Caching Intelligent**
    ```python
    import pickle

    # Save
    with open('embeddings_cache.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    # Load (instantané!)
    with open('embeddings_cache.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    ```
    """)
