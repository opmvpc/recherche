"""
Section TF-IDF pour l'application Streamlit
Contient toutes les fonctions de rendu pour la partie TF-IDF
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Imports from src
from src.tfidf_engine import TFIDFEngine, preprocess_text
from src.data_loader import load_dataset
from src.visualizations import (
    plot_tf_comparison,
    plot_idf_curve,
    plot_idf_wordcloud,
    plot_tfidf_heatmap,
    plot_search_results,
    plot_documents_3d,
    plot_vocabulary_stats,
)


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def render_tab_navigation(tabs_list: list, session_key: str, default_tab: str = None) -> str:
    """
    Rend une navigation par tabs avec des boutons stylés

    Args:
        tabs_list: Liste des noms de tabs
        session_key: Clé de session state pour tracker la tab active
        default_tab: Tab par défaut (optionnel)

    Returns:
        Nom de la tab actuellement sélectionnée
    """
    # Initialiser avec la première tab ou default
    if session_key not in st.session_state:
        st.session_state[session_key] = default_tab if default_tab else tabs_list[0]

    # Rendre les boutons
    cols = st.columns(len(tabs_list))
    for idx, (col, tab_name) in enumerate(zip(cols, tabs_list)):
        with col:
            if st.session_state[session_key] == tab_name:
                # Tab actif - afficher avec style
                st.markdown(
                    f"""
                <div style="
                    background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                    padding: 12px 20px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

    return st.session_state[session_key]


# ============================================================================
# TF-IDF SECTION FUNCTIONS
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
    """)

    # Checkbox pour inclure les datasets étendus
    include_extended = st.checkbox(
        "📦 Inclure les datasets étendus (plus long: ~2-3 minutes)",
        value=False,
        help="Teste aussi les versions étendues des datasets pour voir l'impact sur les performances",
        key="tfidf_bench_extended"
    )

    if include_extended:
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

    if st.button("🚀 Lancer les Benchmarks!", type="primary", key="tfidf_bench_btn"):
        spinner_text = "⏱️ Benchmarking en cours... (2-3 minutes)" if include_extended else "⏱️ Benchmarking en cours... (30 secondes)"

        with st.spinner(spinner_text):
            from src.data_loader import load_dataset
            import time

            # Définir les tests selon le mode
            if include_extended:
                benchmark_tests = [
                    {"name": "recettes", "extended": False, "label": "Recettes (50 docs)"},
                    {"name": "films", "extended": False, "label": "Films (50 docs)"},
                    {"name": "livres", "extended": False, "label": "Livres (100 docs)"},
                    {"name": "recettes", "extended": True, "label": "Recettes étendu (200 docs)"},
                    {"name": "films", "extended": True, "label": "Films étendu (200 docs)"},
                    {"name": "wikipedia", "extended": False, "label": "Wikipedia (100 docs)"},
                    {"name": "livres", "extended": True, "label": "Livres étendu (801 docs)"},
                    {"name": "wikipedia", "extended": True, "label": "Wikipedia étendu (1000 docs)"},
                ]
            else:
                # Mode rapide: seulement les datasets normaux
                benchmark_tests = [
                    {"name": "recettes", "extended": False, "label": "Recettes (50 docs)"},
                    {"name": "films", "extended": False, "label": "Films (50 docs)"},
                    {"name": "livres", "extended": False, "label": "Livres (100 docs)"},
                    {"name": "wikipedia", "extended": False, "label": "Wikipedia (100 docs)"},
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

                    La ligne rouge montre la tendance **linéaire** → confirme la complexité O(n×m)!

                    **Impact de la taille:**
                    - Passer de 50 à 200 docs → ~4× plus lent
                    - Passer de 100 à 1000 docs → ~10× plus lent

                    C'est **proportionnel** au nombre de documents!
                    """)

                st.success("""
                ✅ **Conclusion des Benchmarks:**

                TF-IDF est **rapide et scalable** pour des corpus de taille petite à moyenne!

                - **50-100 docs:** Quasi instantané (< 0.1s) ⚡
                - **200 docs:** Très rapide (< 0.2s) 🚀
                - **800-1000 docs:** Rapide (< 1s) 👌
                - **> 10000 docs:** Optimisations recommandées (index inversé, cache, etc.)

                **💡 À retenir:** La croissance est **linéaire** → prévisible et fiable!
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
