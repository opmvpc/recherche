"""
Section BM25 pour l'application Streamlit
Contient toutes les fonctions de rendu pour la partie BM25
"""

import streamlit as st
import numpy as np
import pandas as pd
import time as time_module  # Renamed to avoid conflict
import matplotlib.pyplot as plt

# Imports from src
from src.bm25_engine import BM25Engine
from src.tfidf_engine import TFIDFEngine
from src.visualizations import (
    plot_search_results,
    plot_saturation_effect,
    plot_length_normalization,
    plot_parameter_space_heatmap,
    plot_tfidf_bm25_comparison,
    plot_score_distributions,
)


# ============================================================================
# HELPER FUNCTION
# ============================================================================


# ============================================================================
# BM25 SECTION FUNCTIONS
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

    # Import de la fonction de navigation stylée
    from app import render_tab_navigation

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
            # Graphique de saturation TF-IDF vs BM25
            fig_saturation = plot_saturation_effect(k1_values=[0.5, 1.2, 1.5, 2.0], max_freq=50)
            st.pyplot(fig_saturation)
            plt.close()

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

        # === GRAPHIQUE COMPARATIF IDF ===
        st.divider()
        st.markdown("### 📊 Visualisation: Impact du Smoothing")

        col_graph, col_analysis = st.columns([3, 2])

        with col_graph:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Nombre total de documents
            N = 1000

            # Gamme de n(q): de 1 à N (nombre de docs contenant le terme)
            n_values = np.arange(1, N + 1)

            # Calcul IDF TF-IDF
            idf_tfidf = np.log(N / n_values)

            # Calcul IDF BM25 (avec smoothing)
            idf_bm25 = np.log((N - n_values + 0.5) / (n_values + 0.5))

            # Tracer les courbes
            ax.plot(
                n_values,
                idf_tfidf,
                "b-",
                linewidth=2.5,
                label="IDF TF-IDF (classique)",
                alpha=0.8,
            )
            ax.plot(
                n_values,
                idf_bm25,
                "r-",
                linewidth=2.5,
                label="IDF BM25 (avec smoothing +0.5)",
                alpha=0.8,
            )

            # Zones d'intérêt
            # Mots très rares (n < 10)
            ax.axvspan(0, 10, alpha=0.1, color="red", label="Mots très rares")
            # Mots communs (n > 800)
            ax.axvspan(800, N, alpha=0.1, color="green", label="Mots très communs")

            # Annotations pour des exemples
            examples_n = [5, 50, 300, 950]
            examples_labels = ["blockchain", "python", "cuisine", "le"]

            for n, label in zip(examples_n, examples_labels):
                idf_tf = np.log(N / n)
                idf_bm = np.log((N - n + 0.5) / (n + 0.5))

                # Marquer sur la courbe TF-IDF
                ax.scatter(
                    [n],
                    [idf_tf],
                    color="blue",
                    s=80,
                    zorder=5,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.text(
                    n,
                    idf_tf + 0.3,
                    f'"{label}"',
                    fontsize=9,
                    ha="center",
                    color="blue",
                    fontweight="bold",
                )

                # Marquer sur la courbe BM25
                ax.scatter(
                    [n],
                    [idf_bm],
                    color="red",
                    s=80,
                    zorder=5,
                    edgecolor="black",
                    linewidth=1.5,
                )

            ax.set_xlabel(
                "Nombre de documents contenant le terme n(q)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel("Score IDF", fontsize=12, fontweight="bold")
            ax.set_title(
                "Comparaison IDF: TF-IDF vs BM25 (Corpus de 1000 docs)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax.legend(fontsize=10, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, N)
            ax.set_ylim(-1, 6)

            # Ligne horizontale à y=0
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_analysis:
            st.markdown("""
            ### 🔍 Analyse du Graphique

            **Courbe bleue (TF-IDF):**
            - Décroît rapidement pour les mots rares
            - **Problème:** Peut devenir négatif si n > N/2 ⚠️
            - Pas de stabilisation

            **Courbe rouge (BM25):**
            - Forme similaire mais **plus stable**
            - Le **+0.5** lisse la courbe
            - Reste toujours positif 💚
            - Évite les valeurs extrêmes

            **Zones colorées:**
            - 🔴 **Rouge:** Mots très rares (< 10 docs)
              - IDF élevé (~5-6)
              - Forte différenciation

            - 🟢 **Vert:** Mots très communs (> 800 docs)
              - IDF proche de 0
              - Faible importance

            **💡 Points clés:**

            Les deux formules donnent des résultats **très similaires** pour la plupart des mots, mais BM25 est plus robuste aux cas extrêmes!

            **Exemple "blockchain" (n=5):**
            - TF-IDF: ~5.30
            - BM25: ~5.30
            - ✅ Quasi identique

            **Avantage BM25:**
            Le smoothing évite les comportements instables pour les mots absents ou extrêmement rares.
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

        # Graphique de saturation
        col_g1, col_g2 = st.columns([3, 2])

        with col_g1:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))

            # Fréquences de 0 à 100
            freqs = np.linspace(0, 100, 200)

            # TF-IDF (linéaire)
            tf_tfidf = freqs
            ax.plot(
                freqs,
                tf_tfidf,
                "r-",
                linewidth=2,
                label="TF-IDF (linéaire)",
                linestyle="--",
                alpha=0.7,
            )

            # BM25 avec différents k1
            k1_vals = [
                (0.5, "#3498db"),
                (1.2, "#e74c3c"),
                (1.5, "#2ecc71"),
                (2.0, "#f39c12"),
            ]
            for k1, color in k1_vals:
                tf_bm25 = (freqs * (k1 + 1)) / (freqs + k1)
                label = f"BM25 (k1={k1})" + (" ⭐" if k1 == 1.5 else "")
                ax.plot(
                    freqs,
                    tf_bm25,
                    color=color,
                    linewidth=2.5 if k1 == 1.5 else 2,
                    label=label,
                )

            ax.set_xlabel("Nombre d'occurrences (f)", fontsize=11)
            ax.set_ylabel("Score TF", fontsize=11)
            ax.set_title(
                "Effet de Saturation: TF-IDF vs BM25", fontsize=12, fontweight="bold"
            )
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 20)

            # Annotations
            ax.axhline(y=1.5, color="green", linestyle=":", alpha=0.5, linewidth=1)
            ax.text(85, 1.7, "Plateau k1=1.5", fontsize=9, color="green")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_g2:
            st.markdown("""
            ### 📈 Analyse du Graphique

            **Axe X:** Nombre d'occurrences
            **Axe Y:** Score TF résultant

            **Ligne rouge pointillée (TF-IDF):**
            - Monte indéfiniment ⬆️
            - 100 occ = score de 100
            - **Problème: spam!** ❌

            **Courbes BM25 (saturées):**
            - **Bleue (k1=0.5):** Plateau ~1.0
            - **Rouge (k1=1.2):** Plateau ~1.2
            - **Verte (k1=1.5) ⭐:** Plateau ~1.5
            - **Orange (k1=2.0):** Plateau ~2.0

            **💡 Observation:**
            Après 20-30 occurrences, les courbes BM25 **plafonnent** → évite la sur-pondération!

            **🎯 Recommandation:**
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

        col_g1, col_g2 = st.columns([3, 2])

        with col_g1:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))

            # Longueurs de documents de 10 à 500 mots
            doc_lengths = np.linspace(10, 500, 200)
            avgdl_val = bm25_demo.avgdl

            # Différentes valeurs de b
            b_vals = [
                (0.0, "#95a5a6"),
                (0.5, "#3498db"),
                (0.75, "#2ecc71"),
                (1.0, "#e74c3c"),
            ]

            for b, color in b_vals:
                norm_factors = 1 - b + b * (doc_lengths / avgdl_val)
                label = f"b={b}" + (" ⭐" if b == 0.75 else "")
                ax.plot(
                    doc_lengths,
                    norm_factors,
                    color=color,
                    linewidth=2.5 if b == 0.75 else 2,
                    label=label,
                )

            # Ligne de référence (facteur = 1)
            ax.axhline(
                y=1.0,
                color="black",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Facteur neutre (1.0)",
            )

            # Ligne verticale à avgdl
            ax.axvline(
                x=avgdl_val,
                color="orange",
                linestyle=":",
                alpha=0.7,
                linewidth=2,
                label=f"avgdl ({avgdl_val:.0f} mots)",
            )

            ax.set_xlabel("Longueur du document (mots)", fontsize=11)
            ax.set_ylabel("Facteur de normalisation", fontsize=11)
            ax.set_title(
                "Effet de Normalisation par Longueur (Paramètre b)",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(10, 500)
            ax.set_ylim(0, 5)

            # Annotations
            ax.text(
                avgdl_val + 20,
                0.2,
                "Longueur moyenne",
                fontsize=9,
                color="orange",
                rotation=0,
            )
            ax.text(
                400,
                0.5,
                "← Docs courts\n   (boost)",
                fontsize=8,
                color="green",
                ha="right",
            )
            ax.text(
                400,
                4.5,
                "← Docs longs\n   (pénalité)",
                fontsize=8,
                color="red",
                ha="right",
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_g2:
            st.markdown(f"""
            ### 📈 Analyse du Graphique

            **Corpus actuel:**
            - avgdl = {bm25_demo.avgdl:.1f} mots

            **Ligne grise (b=0):**
            - Facteur = 1.0 constant
            - **Aucune pénalité**

            **Ligne bleue (b=0.5):**
            - Pénalité modérée

            **Ligne verte (b=0.75) ⭐:**
            - Standard recommandé
            - Équilibre pénalité/boost

            **Ligne rouge (b=1.0):**
            - Pénalité maximale
            - Comme TF-IDF classique

            **💡 Observation:**
            - **Docs < avgdl** → boost (facteur < 1)
            - **Docs > avgdl** → pénalité (facteur > 1)
            - Plus b est élevé, plus l'effet est fort!

            **🎯 Recommandation:**
            b=0.75 pour la plupart des corpus!
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

                col_g1, col_g2 = st.columns([3, 2])  # Plus d'espace pour le graphique

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

    Explore l'impact des paramètres **k1** et **b** sur les scores!

    Cette heatmap montre le **score moyen** des top 5 résultats pour différentes combinaisons de paramètres.
    Plus la zone est rouge/chaude, meilleures sont les séparations de scores!
    """)

    test_query = st.text_input(
        "Query de test:",
        value="recette italienne",
        key="bm25_tuning_query",
        help='💡 Exemples: "plat italien" | "cuisine asiatique" | "dessert chocolat"',
    )

    if test_query:
        with st.spinner(
            "🔥 Calcul de la heatmap pour toutes les combinaisons de paramètres..."
        ):
            # Grille de valeurs à tester
            k1_values = np.linspace(0.5, 3.0, 8)  # 8 valeurs de k1
            b_values = np.linspace(0.0, 1.0, 8)  # 8 valeurs de b

            # Matrice pour stocker les scores moyens
            score_matrix = np.zeros((len(b_values), len(k1_values)))

            # Calculer les scores pour chaque combinaison
            for i, b_val in enumerate(b_values):
                for j, k1_val in enumerate(k1_values):
                    # Créer un engine BM25 avec ces paramètres
                    engine = BM25Engine(
                        documents_texts[:100],  # Limiter à 100 docs pour la rapidité
                        k1=k1_val,
                        b=b_val,
                        remove_stopwords=remove_stopwords,
                    )

                    # Rechercher
                    results = engine.search(test_query, top_k=5)

                    # Score moyen des top 5
                    if results:
                        avg_score = np.mean([score for _, score in results])
                        score_matrix[i, j] = avg_score
                    else:
                        score_matrix[i, j] = 0.0

        # === VISUALISATION HEATMAP ===
        col_heatmap, col_analysis = st.columns([3, 2])

        with col_heatmap:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

            # Heatmap avec annotations
            im = ax.imshow(score_matrix, cmap="YlOrRd", aspect="auto")

            # Axes
            ax.set_xticks(np.arange(len(k1_values)))
            ax.set_yticks(np.arange(len(b_values)))
            ax.set_xticklabels([f"{k1:.2f}" for k1 in k1_values], fontsize=9)
            ax.set_yticklabels([f"{b:.2f}" for b in b_values], fontsize=9)

            # Labels
            ax.set_xlabel("k1 (Saturation du TF)", fontsize=12, fontweight="bold")
            ax.set_ylabel(
                "b (Normalisation de longueur)", fontsize=12, fontweight="bold"
            )
            ax.set_title(
                f'Heatmap BM25: Impact de k1 et b\nQuery: "{test_query}"',
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            # Annotations des valeurs
            for i in range(len(b_values)):
                for j in range(len(k1_values)):
                    ax.text(
                        j,
                        i,
                        f"{score_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

            # Marquer les valeurs standards (k1=1.5, b=0.75)
            k1_std_idx = np.argmin(np.abs(k1_values - 1.5))
            b_std_idx = np.argmin(np.abs(b_values - 0.75))
            ax.add_patch(
                plt.Rectangle(
                    (k1_std_idx - 0.5, b_std_idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=3,
                )
            )
            ax.text(
                k1_std_idx,
                b_std_idx - 0.7,
                "⭐",
                ha="center",
                fontsize=16,
                color="lime",
            )

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Score BM25 moyen (top 5)", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_analysis:
            st.markdown("### 🔍 Analyse de la Heatmap")

            # Trouver les meilleurs paramètres
            best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
            best_b = b_values[best_idx[0]]
            best_k1 = k1_values[best_idx[1]]
            best_score = score_matrix[best_idx]

            # Valeurs standard
            std_score = score_matrix[b_std_idx, k1_std_idx]

            st.markdown(f"""
            **🏆 Meilleure combinaison:**
            - k1 = **{best_k1:.2f}**
            - b = **{best_b:.2f}**
            - Score moyen = **{best_score:.3f}**

            **⭐ Valeurs standard (carré vert):**
            - k1 = **{k1_values[k1_std_idx]:.2f}**
            - b = **{b_values[b_std_idx]:.2f}**
            - Score moyen = **{std_score:.3f}**

            ---

            **📊 Observations:**
            """)

            # Analyses automatiques
            if best_score > std_score * 1.1:
                st.success(f"""
                ✅ **Optimisation possible!**

                Les valeurs optimales donnent {((best_score / std_score - 1) * 100):.1f}% de meilleure séparation que les valeurs standard.
                """)
            else:
                st.info("""
                💡 **Standard = Optimal**

                Les valeurs par défaut (k1=1.5, b=0.75) fonctionnent déjà très bien pour cette query!
                """)

            # Analyse par axe
            avg_by_k1 = np.mean(score_matrix, axis=0)
            avg_by_b = np.mean(score_matrix, axis=1)

            best_k1_overall = k1_values[np.argmax(avg_by_k1)]
            best_b_overall = b_values[np.argmax(avg_by_b)]

            st.markdown(f"""
            **🎯 Recommandations:**

            **Axe k1 (saturation):**
            - Valeur optimale moyenne: **{best_k1_overall:.2f}**
            - {"✅ Saturation faible (< 1.0)" if best_k1_overall < 1.0 else "✅ Saturation modérée (1.0-2.0)" if best_k1_overall < 2.0 else "⚠️ Saturation élevée (> 2.0)"}

            **Axe b (normalisation):**
            - Valeur optimale moyenne: **{best_b_overall:.2f}**
            - {"✅ Pas de normalisation (< 0.3)" if best_b_overall < 0.3 else "✅ Normalisation modérée (0.3-0.8)" if best_b_overall < 0.8 else "⚠️ Forte normalisation (> 0.8)"}
            """)

            st.warning("""
            ⚠️ **Note:**

            Ces résultats dépendent de:
            - La query testée
            - Le corpus utilisé
            - La longueur des documents

            Teste plusieurs queries pour trouver les meilleurs paramètres globaux!
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
        from src.bm25_engine import preprocess_text

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
                if word in mini_bm25.vocabulary:
                    n_t = mini_bm25.doc_freqs.get(word, 0)
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
                    if word in mini_bm25.vocabulary:
                        # IDF
                        n_t = mini_bm25.doc_freqs.get(word, 0)
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
            start_tfidf = time_module.time()
            tfidf_results = tfidf_engine.search(query_compare, top_k=top_k_compare)
            time_tfidf = (time_module.time() - start_tfidf) * 1000  # ms

            start_bm25 = time_module.time()
            bm25_engine = BM25Engine(
                documents_texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
            )
            bm25_results = bm25_engine.search(query_compare, top_k=top_k_compare)
            time_bm25 = (time_module.time() - start_bm25) * 1000  # ms

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
                # Graphique de comparaison TF-IDF vs BM25
                fig_comparison = plot_tfidf_bm25_comparison(
                    tfidf_results,
                    bm25_results,
                    documents_titles,
                    query_compare,
                    top_k=top_k_compare
                )
                st.pyplot(fig_comparison)
                plt.close()

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
                # Histogrammes de distribution des scores
                fig_distributions = plot_score_distributions(tfidf_scores, bm25_scores)
                st.pyplot(fig_distributions)
                plt.close()

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

    # Mesurer le temps de chargement
    start_load = time_module.time()
    n_docs = len(documents_texts)
    total_words = sum(len(doc.split()) for doc in documents_texts)
    avg_length = total_words / n_docs if n_docs > 0 else 0
    time_load = (time_module.time() - start_load) * 1000

    # Mesurer l'indexation BM25
    start_index = time_module.time()
    bm25_engine = BM25Engine(
        documents_texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
    )
    time_index = (time_module.time() - start_index) * 1000

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

    # Checkbox pour inclure les datasets étendus
    include_extended_bm25 = st.checkbox(
        "📦 Inclure les datasets étendus (plus long: ~2-3 minutes)",
        value=False,
        help="Teste aussi les versions étendues des datasets pour voir l'impact sur les performances",
        key="bm25_bench_extended",
    )

    if include_extended_bm25:
        st.warning("""
        ⚠️ **Attention:** Avec les datasets étendus, les benchmarks prendront **2-3 minutes**.

        On testera:
        - 🍝 Recettes: **50 → 200** docs
        - 🎬 Films: **50 → 200** docs
        - 📖 Livres: **100 → 801** docs
        - 📚 Wikipedia: **100 → 1000** docs
        """)
    else:
        st.info("""
        On testera les datasets en mode normal (~30 secondes):
        - 🍝 Recettes: **50** docs
        - 🎬 Films: **50** docs
        - 📖 Livres: **100** docs
        - 📚 Wikipedia: **100** docs
        """)

    if st.button("🚀 Lancer les Benchmarks!", type="primary", key="bm25_bench_btn"):
        spinner_text = (
            "⏳ Benchmarks en cours... (2-3 minutes)"
            if include_extended_bm25
            else "⏳ Benchmarks en cours... (30 secondes)"
        )

        with st.spinner(spinner_text):
            from src.data_loader import load_dataset

            benchmark_results = []

            # Définir les datasets selon le mode
            if include_extended_bm25:
                test_configs = [
                    ("recettes", False, "Recettes (50 docs)"),
                    ("films", False, "Films (50 docs)"),
                    ("livres", False, "Livres (100 docs)"),
                    ("recettes", True, "Recettes étendu (200 docs)"),
                    ("films", True, "Films étendu (200 docs)"),
                    ("wikipedia", False, "Wikipedia (100 docs)"),
                    ("livres", True, "Livres étendu (801 docs)"),
                    ("wikipedia", True, "Wikipedia étendu (1000 docs)"),
                ]
            else:
                # Mode rapide: seulement les datasets normaux
                test_configs = [
                    ("recettes", False, "Recettes (50 docs)"),
                    ("films", False, "Films (50 docs)"),
                    ("livres", False, "Livres (100 docs)"),
                    ("wikipedia", False, "Wikipedia (100 docs)"),
                ]

            for dataset_name, extended, label in test_configs:
                try:
                    # Charger le dataset
                    start = time_module.time()
                    dataset = load_dataset(dataset_name, extended=extended)
                    time_load_bench = (time_module.time() - start) * 1000

                    if len(dataset) == 0:
                        continue

                    texts = [doc["text"] for doc in dataset]
                    n_bench = len(texts)

                    # Indexation BM25
                    start = time_module.time()
                    bm25_bench = BM25Engine(
                        texts, k1=1.5, b=0.75, remove_stopwords=remove_stopwords
                    )
                    time_index_bench = (time_module.time() - start) * 1000

                    # Recherche test
                    test_query = "test recherche exemple"
                    start = time_module.time()
                    _ = bm25_bench.search(test_query, top_k=5)
                    time_search = (time_module.time() - start) * 1000

                    vocab_bench = len(bm25_bench.vocabulary)

                    benchmark_results.append(
                        {
                            "Dataset": label,
                            "Docs": n_bench,
                            "Vocab": vocab_bench,
                            "Load (s)": f"{time_load_bench / 1000:.3f}",
                            "Index (s)": f"{time_index_bench / 1000:.3f}",
                            "Search (s)": f"{time_search / 1000:.3f}",
                            "Total (s)": f"{(time_load_bench + time_index_bench) / 1000:.3f}",
                            "_total_numeric": (time_load_bench + time_index_bench)
                            / 1000,
                            "_docs_numeric": n_bench,
                        }
                    )

                except Exception as e:
                    st.warning(f"⚠️ Erreur avec {dataset_name}: {str(e)}")
                    continue

            if len(benchmark_results) > 0:
                # Afficher les résultats
                st.markdown("### 📊 Résultats des Benchmarks")

                df_bench = pd.DataFrame(benchmark_results)
                df_display = df_bench.drop(columns=["_total_numeric", "_docs_numeric"])
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Graphique: Temps vs Nombre de docs (style TF-IDF)
                st.markdown(
                    "### 📈 Graphique: Temps d'Indexation vs Nombre de Documents"
                )

                col_graph, col_analysis = st.columns([2, 1])

                with col_graph:
                    import matplotlib.pyplot as plt

                    x = [r["_docs_numeric"] for r in benchmark_results]
                    y = [r["_total_numeric"] for r in benchmark_results]
                    labels = [r["Dataset"] for r in benchmark_results]

                    fig, ax = plt.subplots(figsize=(8, 5))

                    # Scatter plot
                    ax.scatter(x, y, s=100, alpha=0.6, color="#2ca02c")

                    # Labels pour chaque point
                    for i, label in enumerate(labels):
                        ax.annotate(
                            label.split("(")[0].strip(),
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
                    ax.set_title("Performance BM25: Temps vs Taille du Corpus")
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                with col_analysis:
                    st.markdown("**🔍 Analyse:**")

                    fastest = min(benchmark_results, key=lambda x: x["_total_numeric"])
                    slowest = max(benchmark_results, key=lambda x: x["_total_numeric"])

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

                    La ligne rouge montre la tendance **linéaire** → confirme la complexité O(n×m)!

                    **Impact de la taille:**
                    - Passer de 50 à 200 docs → ~4× plus lent
                    - Passer de 100 à 1000 docs → ~10× plus lent

                    C'est **proportionnel** au nombre de documents!
                    """)

                st.success("""
                ✅ **Conclusion des Benchmarks:**

                BM25 est **rapide et scalable** pour des corpus de taille petite à moyenne!

                - **50-100 docs:** Quasi instantané (< 0.1s) ⚡
                - **200 docs:** Très rapide (< 0.2s) 🚀
                - **800-1000 docs:** Rapide (< 1s) 👌
                - **> 10000 docs:** Optimisations recommandées (index inversé, cache, etc.)

                **💡 À retenir:** La croissance est **linéaire** → prévisible et fiable!

                **🆚 Comparaison avec TF-IDF:**
                - Légèrement plus lent (normalisation par longueur)
                - Mais **meilleurs résultats** sur des docs de longueurs variées!
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
            start = time_module.time()
            tfidf_engine = TFIDFEngine(
                documents_texts[:100], remove_stopwords=remove_stopwords
            )
            tfidf_engine.fit()
            tfidf_time = time_module.time() - start

            # BM25
            start = time_module.time()
            bm25_engine = BM25Engine(
                documents_texts[:100], remove_stopwords=remove_stopwords
            )
            bm25_time = time_module.time() - start

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
