"""
Section Synthèse Comparative pour l'application
À intégrer dans app.py principal
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Imports des visualizations nécessaires
from visualizations import plot_technique_comparison_radar


# ============================================================================
# SECTION SYNTHÈSE COMPARATIVE COMPLÈTE
# ============================================================================

def render_synthesis_section(dataset, documents_texts, documents_titles, documents_categories,
                             tfidf_engine, bm25_engine, embedding_engine):
    """Section Synthèse complète"""

    st.title("📊 Synthèse: Quelle Technique Choisir?")

    st.markdown("""
    Tu as exploré **3 techniques de recherche textuelle**.
    Maintenant, découvre **quand** et **pourquoi** utiliser chacune! 🎯
    """)

    # Import de la fonction de navigation stylée
    from app import render_tab_navigation

    # Sub-navigation avec beaux boutons (style cohérent avec autres sections)
    tabs_list = [
        "📋 Tableau Comparatif",
        "🎯 Guide Décision",
        "💼 Cas d'Usage",
        "🔬 Benchmark",
        "🚀 Recommandations",
    ]
    tab = render_tab_navigation(
        tabs_list, "synthesis_current_tab", default_tab="📋 Tableau Comparatif"
    )

    # Afficher la sous-section correspondante
    if tab == "📋 Tableau Comparatif":
        render_synthesis_comparison_table()
    elif tab == "🎯 Guide Décision":
        render_synthesis_decision_guide()
    elif tab == "💼 Cas d'Usage":
        render_synthesis_use_cases()
    elif tab == "🔬 Benchmark":
        render_synthesis_benchmark(tfidf_engine, bm25_engine, embedding_engine, documents_texts, documents_titles)
    elif tab == "🚀 Recommandations":
        render_synthesis_recommendations()


def render_synthesis_comparison_table():
    """Tableau comparatif complet des 3 techniques"""
    st.header("📋 Tableau Comparatif Complet")

    st.markdown("""
    Comparaison exhaustive de **TF-IDF**, **BM25**, **Embeddings**, et **Hybrid** sur tous les critères!
    """)

    # Tableau interactif
    comparison_data = {
        'Critère': [
            'Type de matching',
            'Synonymes',
            'Typos/Fautes',
            'Noms propres',
            'Codes/IDs',
            'Polysémie (contexte)',
            'Relations conceptuelles',
            'Vitesse indexation',
            'Vitesse recherche',
            'Mémoire requise',
            'Ressources (CPU/GPU)',
            'Multilingue',
            'Interprétabilité',
            'Scalabilité',
            'Facilité implémentation',
            'Coût infrastructure'
        ],
        'TF-IDF': [
            'Lexical (mots exacts)',
            '❌ Fail complet',
            '❌ Fail complet',
            '✅ Excellent',
            '✅ Excellent',
            '❌ Pas de distinction',
            '❌ Aucune',
            '⚡⚡⚡ Très rapide',
            '⚡⚡⚡ <5ms',
            '💾 Minimal (~2MB/1k docs)',
            '💻 CPU uniquement',
            '❌ 1 langue',
            '✅✅✅ Très clair',
            '✅✅✅ Excellent',
            '✅✅✅ Facile',
            '💰 Minimal'
        ],
        'BM25': [
            'Lexical (mots exacts)',
            '❌ Fail complet',
            '❌ Fail complet',
            '✅ Excellent',
            '✅ Excellent',
            '❌ Pas de distinction',
            '❌ Aucune',
            '⚡⚡⚡ Très rapide',
            '⚡⚡⚡ <5ms',
            '💾 Minimal (~2MB/1k docs)',
            '💻 CPU uniquement',
            '❌ 1 langue',
            '✅✅✅ Très clair',
            '✅✅✅ Excellent',
            '✅✅ Facile',
            '💰 Minimal'
        ],
        'Embeddings': [
            'Sémantique (sens)',
            '✅✅✅ Excellent',
            '⚠️ Partiel',
            '⚠️ Moyen',
            '❌ Faible',
            '✅✅ Bon',
            '✅✅✅ Excellent',
            '⚡ Lent (~30s GPU)',
            '⚡⚡ ~10ms',
            '💾💾 Moyen (~15MB/1k docs)',
            '🎮 GPU recommandé',
            '✅✅✅ Multi-langue',
            '❌ Boîte noire',
            '⚠️ Coûteux (>10k docs)',
            '⚠️ Complexe',
            '💰💰 Moyen-élevé'
        ],
        'Hybrid': [
            'Lexical + Sémantique',
            '✅✅✅ Excellent',
            '⚠️ Partiel',
            '✅✅ Très bon',
            '✅✅ Très bon',
            '✅✅ Bon',
            '✅✅✅ Excellent',
            '⚡ Lent (~30s GPU)',
            '⚡⚡ ~15ms',
            '💾💾 Moyen',
            '💻 + 🎮',
            '✅✅✅ Multi-langue',
            '⚠️ Partiel',
            '✅✅ Bon',
            '⚠️ Complexe',
            '💰💰 Moyen-élevé'
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)

    st.dataframe(
        df_comparison.set_index('Critère'),
        use_container_width=True,
        height=700
    )

    st.caption("""
    **Légende:**
    ✅ = Bon | ⚠️ = Moyen | ❌ = Faible
    ⚡ = Rapide | 💾 = Mémoire | 💻 = CPU | 🎮 = GPU | 💰 = Coût
    """)

    # Insights
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **✅ Points Forts par Technique:**

        **TF-IDF:**
        - Simplicité et rapidité
        - Noms propres et codes
        - Infrastructure minimale

        **BM25:**
        - Meilleur que TF-IDF partout
        - Toujours le bon choix si lexical

        **Embeddings:**
        - Synonymes et concepts
        - Multi-langue natif
        - Recherche sémantique puissante

        **Hybrid:**
        - Meilleur des deux mondes
        - Flexibilité maximale
        """)

    with col2:
        st.warning("""
        **⚠️ Limitations par Technique:**

        **TF-IDF:**
        - Aucune sémantique
        - Obsolète (utiliser BM25)

        **BM25:**
        - Pas de synonymes
        - 1 seule langue

        **Embeddings:**
        - Lent à indexer
        - Coûteux (GPU)
        - Faible sur codes/IDs

        **Hybrid:**
        - Complexité d'implémentation
        - Tuning du paramètre α
        """)


def render_synthesis_decision_guide():
    """Arbre de décision interactif - VERSION PÉDAGOGIQUE"""
    st.header("🎯 Guide de Décision Interactif")

    st.info("""
    **💡 Objectif de cette section:**

    Te guider vers la MEILLEURE technique pour TON cas d'usage spécifique!

    Réponds honnêtement aux questions en pensant à TON application réelle.
    """)

    st.markdown("""
    ### 🌳 Quel Technique Pour Ton Cas?

    Réponds à ces questions pour trouver la meilleure solution:
    """)

    # Quiz interactif
    q1 = st.radio(
        "**1. Ton corpus contient principalement:**",
        [
            "Du texte naturel (articles, descriptions)",
            "Des données structurées (codes, IDs, noms)",
            "Un mélange des deux"
        ],
        key="quiz_q1"
    )

    q2 = st.radio(
        "**2. Tes utilisateurs recherchent plutôt par:**",
        [
            "Mots-clés exacts",
            "Concepts/idées (synonymes OK)",
            "Les deux"
        ],
        key="quiz_q2"
    )

    q3 = st.radio(
        "**3. Tes contraintes de performance:**",
        [
            "Temps réel critique (<10ms)",
            "Performance importante mais flexible",
            "Pas de contrainte forte"
        ],
        key="quiz_q3"
    )

    q4 = st.radio(
        "**4. Ton budget infrastructure:**",
        [
            "Minimal (pas de GPU)",
            "Moyen (GPU possible)",
            "Flexible"
        ],
        key="quiz_q4"
    )

    q5 = st.radio(
        "**5. Multilingue nécessaire?**",
        ["Oui", "Non"],
        key="quiz_q5"
    )

    if st.button("🎯 Voir la recommandation!", type="primary", key="quiz_submit"):
        # Logique de décision
        score_tfidf = 0
        score_bm25 = 0
        score_embeddings = 0
        score_hybrid = 0

        # Q1
        if "structurées" in q1:
            score_bm25 += 3
            score_tfidf += 2
        elif "naturel" in q1:
            score_embeddings += 3
            score_hybrid += 2
        else:
            score_hybrid += 3
            score_bm25 += 2

        # Q2
        if "exacts" in q2:
            score_bm25 += 3
            score_tfidf += 2
        elif "Concepts" in q2:
            score_embeddings += 3
            score_hybrid += 1
        else:
            score_hybrid += 3

        # Q3
        if "réel" in q3:
            score_bm25 += 3
            score_tfidf += 3
        elif "importante" in q3:
            score_bm25 += 2
            score_hybrid += 1
        else:
            score_embeddings += 2
            score_hybrid += 2

        # Q4
        if "Minimal" in q4:
            score_bm25 += 3
            score_tfidf += 3
        elif "Moyen" in q4:
            score_hybrid += 2
            score_embeddings += 1
        else:
            score_embeddings += 2
            score_hybrid += 2

        # Q5
        if q5 == "Oui":
            score_embeddings += 3
            score_hybrid += 2
        else:
            score_bm25 += 1

        # Recommandation
        scores = {
            'TF-IDF': score_tfidf,
            'BM25': score_bm25,
            'Embeddings': score_embeddings,
            'Hybrid': score_hybrid
        }

        best_technique = max(scores, key=scores.get)

        # Affichage
        st.markdown("### 🏆 Recommandation")

        if best_technique == 'BM25':
            st.success(f"""
            **🎯 BM25 est recommandé pour ton cas!**

            **Pourquoi?**
            - Recherche lexicale efficace
            - Performance excellente
            - Infrastructure minimale
            - Facile à implémenter

            **Conseil:** Commence avec BM25, ajoute Embeddings plus tard si besoin.
            """)
        elif best_technique == 'Embeddings':
            st.success(f"""
            **🧠 Embeddings sont recommandés pour ton cas!**

            **Pourquoi?**
            - Recherche sémantique puissante
            - Multi-langue natif
            - Comprend les concepts

            **Conseil:** Investis dans un GPU pour de bonnes performances.
            """)
        elif best_technique == 'Hybrid':
            st.success(f"""
            **🎨 Hybrid Search (BM25 + Embeddings) est recommandé!**

            **Pourquoi?**
            - Meilleur des deux mondes
            - Flexibilité maximale
            - Qualité optimale

            **Conseil:** Commence avec α=0.5, ajuste selon tes métriques.
            """)
        else:
            st.success(f"""
            **📊 TF-IDF est recommandé pour ton cas!**

            **Pourquoi?**
            - Simple et efficace
            - Ressources minimales
            - Bon point de départ

            **Conseil:** Migre vers BM25 pour de meilleures performances.
            """)

        # Scores détaillés
        with st.expander("📊 Voir les scores détaillés"):
            scores_df = pd.DataFrame({
                'Technique': list(scores.keys()),
                'Score': list(scores.values())
            }).sort_values('Score', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(scores_df['Technique'], scores_df['Score'],
                          color=['green' if i == 0 else 'lightblue' for i in range(len(scores_df))])
            ax.set_xlabel('Score de Pertinence', fontweight='bold')
            ax.set_title('Scores par Technique', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for bar, score in zip(bars, scores_df['Score']):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f' {score}',
                       va='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)


def render_synthesis_use_cases():
    """Cas d'usage réels par industrie"""
    st.header("💼 Cas d'Usage Réels par Industrie")

    st.markdown("""
    Découvre comment choisir la bonne technique selon ton **domaine d'application**!
    """)

    # Exemples concrets par industrie
    use_cases = {
        '🛒 E-commerce': {
            'description': "Recherche de produits avec filtres",
            'challenges': [
                "Synonymes produits (\"téléphone\" = \"smartphone\")",
                "Variantes (\"Nike Air\" vs \"Air Nike\")",
                "Filtres exacts (marque, prix)"
            ],
            'recommendation': '🎨 Hybrid',
            'config': "BM25 (40%) + Embeddings (60%)",
            'justification': "Combine matching exact de marques avec compréhension sémantique",
            'code_lang': 'python',
            'code': """from src.hybrid_search import HybridSearch

# Configuration
hybrid = HybridSearch(
    documents,
    bm25_engine,
    embedding_engine,
    alpha=0.4  # 40% BM25, 60% Embeddings
)

# Recherche
results = hybrid.search("query", top_k=10)"""
        },
        '📚 Documentation Technique': {
            'description': "Recherche dans docs d'API, code",
            'challenges': [
                "Noms de fonctions exacts",
                "Code snippets",
                "Erreurs techniques"
            ],
            'recommendation': '🎯 BM25',
            'config': "k1=1.2, b=0.75",
            'justification': "Priorité au matching exact pour les termes techniques",
            'code_lang': 'python',
            'code': """from src.bm25_engine import BM25Engine

# Configuration
engine = BM25Engine(
    documents,
    k1=1.2,
    b=0.75
)

# Recherche
results = engine.search("query", top_k=10)"""
        },
        '💬 Support Client / FAQ': {
            'description': "Trouver réponses aux questions clients",
            'challenges': [
                "Multiples formulations même question",
                "Synonymes fréquents",
                "Questions complexes"
            ],
            'recommendation': '🧠 Embeddings',
            'config': "Sentence-BERT multilingue",
            'justification': "Comprend l'intention derrière différentes formulations",
            'code_lang': 'python',
            'code': """from sentence_transformers import SentenceTransformer

# Modèle multilingue
model = SentenceTransformer(
    'paraphrase-multilingual-mpnet-base-v2'
)

# Index
embeddings = model.encode(documents)

# Recherche
query_emb = model.encode([query])
similarities = cosine_similarity(query_emb, embeddings)"""
        },
        '📰 Recherche d\'Articles': {
            'description': "Moteur de recherche de contenu",
            'challenges': [
                "Concepts similaires",
                "Recherche par thème",
                "Multi-langues"
            ],
            'recommendation': '🎨 Hybrid',
            'config': "BM25 (30%) + Embeddings (70%)",
            'justification': "Sémantique pour concepts, lexical pour noms propres",
            'code_lang': 'python',
            'code': """hybrid = HybridSearch(
    documents,
    bm25_engine,
    embedding_engine,
    alpha=0.3  # Plus de poids sur sémantique
)"""
        },
        '🏥 Dossiers Médicaux': {
            'description': "Recherche dans historiques patients",
            'challenges': [
                "Termes médicaux exacts",
                "IDs patients/médicaments",
                "Réglementation (traçabilité)"
            ],
            'recommendation': '🎯 BM25',
            'config': "k1=1.5, b=0.75 + index inversé",
            'justification': "Interprétabilité et matching exact requis",
            'code_lang': 'python',
            'code': """# BM25 avec index inversé pour performance
engine = BM25Engine(documents, k1=1.5, b=0.75)"""
        },
        '🎓 Plateforme Éducative': {
            'description': "Recherche de cours, ressources",
            'challenges': [
                "Concepts pédagogiques",
                "Multi-niveaux",
                "Recommandations"
            ],
            'recommendation': '🧠 Embeddings',
            'config': "Embeddings + clustering thématique",
            'justification': "Recommandations sémantiques de contenu similaire",
            'code_lang': 'python',
            'code': """# Embeddings pour recommendations
embedding_engine = EmbeddingSearch()
embedding_engine.index(courses)

# Clustering automatique
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(embeddings)"""
        }
    }

    # Sélection
    selected_industry = st.selectbox(
        "Choisis un domaine:",
        list(use_cases.keys()),
        key="industry_select"
    )

    use_case = use_cases[selected_industry]

    # Affichage détaillé
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {selected_industry}")
        st.write(use_case['description'])

        st.markdown("**Challenges:**")
        for challenge in use_case['challenges']:
            st.write(f"- {challenge}")

    with col2:
        st.metric("✨ Recommandation", use_case['recommendation'])
        st.caption(use_case['config'])

    st.info(f"**💡 Justification:** {use_case['justification']}")

    # Exemple de code
    with st.expander("💻 Exemple d'implémentation"):
        st.code(use_case['code'], language=use_case['code_lang'])


def render_synthesis_benchmark(tfidf_engine, bm25_engine, embedding_engine, documents_texts, documents_titles):
    """Benchmark comparatif des techniques"""
    st.header("🔬 Benchmark: Qualité vs Performance")

    st.markdown("""
    ### Métriques de Qualité

    Pour évaluer les techniques, on utilise:
    - **Precision@K:** % de résultats pertinents dans les top K
    - **Recall@K:** % de documents pertinents trouvés
    - **MRR (Mean Reciprocal Rank):** Position moyenne du 1er résultat pertinent
    - **NDCG:** Mesure tenant compte du ranking
    """)

    # Résultats simulés (basés sur littérature)
    benchmark_results = {
        'Métrique': ['Precision@10', 'Recall@10', 'MRR', 'NDCG@10', 'Temps (ms)'],
        'TF-IDF': [0.45, 0.52, 0.38, 0.51, 3],
        'BM25': [0.58, 0.64, 0.52, 0.63, 4],
        'Embeddings': [0.76, 0.81, 0.71, 0.79, 12],
        'Hybrid': [0.82, 0.86, 0.78, 0.84, 16]
    }

    df_benchmark = pd.DataFrame(benchmark_results)

    st.dataframe(df_benchmark, use_container_width=True)

    # Graphique radar
    st.markdown("### 📊 Visualisation Radar")

    metrics = {
        'TF-IDF': {'Precision@10': 0.45, 'Recall@10': 0.52, 'MRR': 0.38, 'NDCG@10': 0.51},
        'BM25': {'Precision@10': 0.58, 'Recall@10': 0.64, 'MRR': 0.52, 'NDCG@10': 0.63},
        'Embeddings': {'Precision@10': 0.76, 'Recall@10': 0.81, 'MRR': 0.71, 'NDCG@10': 0.79},
        'Hybrid': {'Precision@10': 0.82, 'Recall@10': 0.86, 'MRR': 0.78, 'NDCG@10': 0.84}
    }

    fig_radar = plot_technique_comparison_radar(metrics)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    **📊 Interprétation:**

    - **Embeddings:** Meilleure qualité, mais plus lent
    - **BM25:** Bon compromis qualité/vitesse
    - **Hybrid:** Meilleur qualité globale
    - **TF-IDF:** Base de comparaison (baseline)
    """)


def render_synthesis_recommendations():
    """Recommandations finales et feuille de route"""
    st.header("🚀 Feuille de Route Recommandée")

    st.markdown("""
    ### 🛤️ Parcours d'Adoption Progressif

    Voici comment implémenter la recherche selon ta maturité:
    """)

    # Timeline
    timeline_data = {
        'Phase 1 (Semaine 1)': {
            'technique': 'BM25',
            'objectif': 'MVP fonctionnel',
            'actions': [
                'Implémenter BM25 basique',
                'Indexer ton corpus',
                'Interface de recherche simple',
                'Métriques de base (latence, nb résultats)'
            ],
            'effort': '⚡ 1-2 jours',
            'coût': '💰 Minimal'
        },
        'Phase 2 (Semaine 2-3)': {
            'technique': 'BM25 Optimisé',
            'objectif': 'Production-ready',
            'actions': [
                'Tuning k1/b selon métriques',
                'Index inversé pour performance',
                'Filtres (date, catégorie, etc.)',
                'Logging & monitoring'
            ],
            'effort': '⚡⚡ 3-5 jours',
            'coût': '💰 Minimal'
        },
        'Phase 3 (Mois 2)': {
            'technique': 'Embeddings (Pilot)',
            'objectif': 'Test sémantique',
            'actions': [
                'Setup Sentence-BERT',
                'Indexer subset corpus (10-20%)',
                'A/B test vs BM25',
                'Mesurer impact qualité'
            ],
            'effort': '⚡⚡⚡ 1-2 semaines',
            'coût': '💰💰 Moyen (GPU)'
        },
        'Phase 4 (Mois 3)': {
            'technique': 'Hybrid',
            'objectif': 'Qualité optimale',
            'actions': [
                'Combiner BM25 + Embeddings',
                'Tuning α selon feedback',
                'Full corpus embeddings',
                'FAISS pour performance'
            ],
            'effort': '⚡⚡⚡ 2-3 semaines',
            'coût': '💰💰 Moyen'
        }
    }

    # Affichage chronologique
    for phase, data in timeline_data.items():
        with st.expander(f"**{phase}: {data['technique']}** - {data['objectif']}"):
            st.markdown(f"**Actions:**")
            for action in data['actions']:
                st.write(f"✓ {action}")

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Effort: {data['effort']}")
            with col2:
                st.caption(f"Coût: {data['coût']}")

    # Conseils finaux
    st.markdown("---")
    st.markdown("### 💡 Conseils Pratiques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ À FAIRE:**

        - ✅ Commencer simple (BM25)
        - ✅ Mesurer avant d'optimiser
        - ✅ A/B tester les changements
        - ✅ Écouter les utilisateurs
        - ✅ Documenter les choix
        - ✅ Monitorer la performance
        """)

    with col2:
        st.markdown("""
        **❌ À ÉVITER:**

        - ❌ Over-engineering initial
        - ❌ Embeddings sans GPU
        - ❌ Ignorer BM25 (toujours utile!)
        - ❌ Oublier le monitoring
        - ❌ Tuning sans métriques
        - ❌ Négliger l'UX
        """)

    # Call to action final
    st.success("""
    ### 🎯 Prochaines Étapes

    1. **Expérimente** avec les 3 techniques sur ton corpus
    2. **Mesure** la qualité avec tes utilisateurs
    3. **Itère** selon les retours
    4. **Scale** progressivement

    **Bonne chance dans tes projets de recherche! 🚀**
    """)

    st.balloons()


def render_synthesis_recommendations():
    """Section Recommandations pour le projet (déplacée depuis Embeddings)"""
    st.header("🚀 Recommandations pour Votre Projet")

    st.markdown("""
    Vous êtes en train de développer votre projet web! Voici comment **intégrer des embeddings dans votre application** de manière professionnelle et économique. 💼
    """)

    st.divider()

    # === OPENROUTER ===
    st.markdown("## 🌐 Option 1: API Embeddings avec OpenRouter")

    st.markdown("""
    **Pourquoi OpenRouter?**
    - 🚀 **Rapide:** Exécution sur GPU professionnel
    - 💰 **Pas cher:** À partir de $0.005/M tokens
    - 🎯 **Qualité supérieure:** Modèles optimisés et maintenus
    - 🔧 **Simple:** API REST standard, pas de setup serveur
    - 🌍 **Scalable:** Gère automatiquement la montée en charge
    """)

    st.markdown("### 📊 Modèles Disponibles (Sélection)")

    # Tableau des modèles avec prix
    models_data = [
        {
            "Modèle": "all-MiniLM-L6-v2",
            "Dimensions": 384,
            "Contexte": "512 tokens",
            "Prix": "$0.005/M",
            "Cas d'usage": "Léger, rapide",
        },
        {
            "Modèle": "all-MiniLM-L12-v2",
            "Dimensions": 384,
            "Contexte": "512 tokens",
            "Prix": "$0.005/M",
            "Cas d'usage": "Équilibré",
        },
        {
            "Modèle": "all-mpnet-base-v2",
            "Dimensions": 768,
            "Contexte": "512 tokens",
            "Prix": "$0.005/M",
            "Cas d'usage": "Haute qualité",
        },
        {
            "Modèle": "bge-base-en-v1.5",
            "Dimensions": 768,
            "Contexte": "512 tokens",
            "Prix": "$0.005/M",
            "Cas d'usage": "Retrieval",
        },
        {
            "Modèle": "multilingual-e5-large",
            "Dimensions": 1024,
            "Contexte": "512 tokens",
            "Prix": "$0.01/M",
            "Cas d'usage": "Multilingue",
        },
        {
            "Modèle": "bge-m3",
            "Dimensions": 1024,
            "Contexte": "8K tokens",
            "Prix": "$0.01/M",
            "Cas d'usage": "Longs docs",
        },
        {
            "Modèle": "OpenAI ada-002",
            "Dimensions": 1536,
            "Contexte": "8K tokens",
            "Prix": "$0.10/M",
            "Cas d'usage": "Référence",
        },
        {
            "Modèle": "OpenAI text-3-large",
            "Dimensions": 3072,
            "Contexte": "8K tokens",
            "Prix": "$0.13/M",
            "Cas d'usage": "Top qualité",
        },
    ]

    import pandas as pd

    df_models = pd.DataFrame(models_data)
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.divider()

    # === CALCUL DE COÛT ===
    st.markdown("### 💰 Calcul de Coût: C'est VRAIMENT Pas Cher!")

    st.markdown("""
    **Exemple concret:** Embedder **TOUS les livres Harry Potter + La Bible**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📚 Corpus complet", "~2.5 millions de mots")
        st.metric("🔢 Tokens (approx)", "~3.3M tokens")
        st.metric("📊 Nombre de documents", "~10,000 chunks")

    with col2:
        st.metric("💵 Coût (MiniLM)", "$0.016")
        st.metric("💵 Coût (MPNet)", "$0.016")
        st.metric("💵 Coût (E5-Large)", "$0.033")

    st.success("""
    ✅ **Résultat:** Même les 7 livres Harry Potter + la Bible = **moins de 5 centimes**!

    Pour votre projet étudiant avec quelques centaines/milliers de documents: **~$0.001 à $0.01**
    → Moins qu'un café! ☕
    """)

    st.info("""
    💡 **Astuce:**
    - Embedder **une seule fois** au début (indexation)
    - Stocker les embeddings dans votre DB
    - Embedder **seulement la query** à chaque recherche (~0.0001¢ par recherche)

    **Coût réel en prod:** Négligeable! 🎉
    """)

    st.divider()

    # === COMPARAISON LOCAL VS API ===
    st.markdown("### ⚖️ Local vs API: Comparaison")

    comparison_data = {
        "Critère": [
            "Qualité",
            "Vitesse",
            "Setup",
            "Coût initial",
            "Coût usage",
            "Maintenance",
            "GPU requis",
            "Scalabilité",
        ],
        "Local (CPU)": [
            "✅ Bon",
            "🐌 Lent (1-10s)",
            "😰 Complexe",
            "💰 Gratuit",
            "💵 Électricité",
            "🔧 À faire",
            "❌ Non",
            "⚠️ Limitée",
        ],
        "Local (GPU)": [
            "✅✅ Excellent",
            "⚡ Rapide (0.1s)",
            "😱 Très complexe",
            "💰💰💰 Cher",
            "💵💵 Électricité",
            "🔧🔧 Maintenance",
            "✅ Oui (CUDA)",
            "⚠️ Limitée",
        ],
        "API (OpenRouter)": [
            "✅✅ Excellent",
            "⚡⚡ Très rapide",
            "😊 Simple",
            "💰 Gratuit",
            "💵 ~$0.005/M",
            "✨ Aucune",
            "☁️ Géré",
            "🚀 Illimitée",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.divider()

    # === OUTILS ET PLATEFORMES ===
    st.markdown("## 🛠️ Outils et Plateformes Recommandés")

    col_tool1, col_tool2 = st.columns(2)

    with col_tool1:
        st.markdown("""
        ### 🗄️ Vector Databases

        **LanceDB** 🏆
        - 📦 Self-hosted ou cloud
        - 🚀 Ultra rapide (Rust)
        - 💾 Fichiers locaux ou S3
        - 🐍 API Python simple
        - 💰 Gratuit (self-hosted)

        ```python
        import lancedb

        db = lancedb.connect("./data/vectors")
        table = db.create_table("docs",
                                data=embeddings)

        # Recherche
        results = table.search(query_vec)
                      .limit(10)
                      .to_list()
        ```
        """)

    with col_tool2:
        st.markdown("""
        ### 🐘 PostgreSQL + pgvector

        **Extension pgvector** 🎯
        - 🗄️ DB que vous connaissez déjà!
        - 🔧 Extension simple à installer
        - 💼 Production-ready
        - 🔗 Combine vecteurs + données SQL

        ```sql
        CREATE EXTENSION vector;

        CREATE TABLE documents (
          id SERIAL PRIMARY KEY,
          content TEXT,
          embedding vector(384)
        );

        -- Recherche par similarité
        SELECT * FROM documents
        ORDER BY embedding <-> $1
        LIMIT 10;
        ```
        """)

    st.markdown("""
    ### 📚 Autres options populaires:
    - **Pinecone:** Cloud, très simple, gratuit jusqu'à 1M vecteurs
    - **Weaviate:** Open-source, features riches (filtres, hybrid search)
    - **Qdrant:** Rust, performant, filtres avancés
    - **Milvus:** Enterprise-grade, très scalable
    """)

    st.divider()

    # === CAS D'USAGE POUR LE PROJET ===
    st.markdown("## 💼 Cas d'Usage pour Votre Projet")

    use_cases = [
        {
            "icon": "💬",
            "title": "Recherche dans Conversations",
            "desc": "Retrouver des messages par sens (pas seulement mots-clés)",
            "example": "Query: 'bug de connexion' → Trouve: 'je peux plus me logger'",
        },
        {
            "icon": "📄",
            "title": "Base de Docs/Knowledge Base",
            "desc": "Recherche sémantique dans documentation, FAQs, wikis",
            "example": "Query: 'comment reset mdp?' → Trouve docs sur réinitialisation",
        },
        {
            "icon": "🛍️",
            "title": "Recherche Produits E-commerce",
            "desc": "Recommandations basées sur descriptions similaires",
            "example": "Query: 'chaussures confort été' → Trouve sandales légères",
        },
        {
            "icon": "📧",
            "title": "Emails / Tickets Support",
            "desc": "Classer et retrouver tickets similaires automatiquement",
            "example": "Nouveau ticket → Suggère solutions de tickets similaires passés",
        },
        {
            "icon": "📰",
            "title": "Articles / Blog",
            "desc": "Recommander articles similaires, clustering de contenus",
            "example": "'Articles liés' basés sur vraie similarité de contenu",
        },
    ]

    for uc in use_cases:
        with st.expander(f"{uc['icon']} {uc['title']}"):
            st.markdown(f"**Description:** {uc['desc']}")
            st.info(f"**Exemple:** {uc['example']}")

    st.divider()

    # === RAG (TEASER) ===
    st.markdown("## 🎓 Et Après? RAG (Retrieval-Augmented Generation)")

    st.markdown("""
    Les embeddings sont la **base des systèmes RAG** (Retrieval-Augmented Generation):

    **Principe:**
    1. 🔍 **Recherche** (embeddings) → Trouver les docs pertinents
    2. 📄 **Context** → Injecter ces docs dans le prompt
    3. 🤖 **Generation** (LLM) → Générer une réponse basée sur VOS données

    **Exemple:**
    ```
    User: "Quelle est notre politique de remboursement?"

    → [Embeddings] Trouve les 3 docs les plus pertinents
    → [LLM] Génère réponse basée sur CES docs (pas hallucination!)
    ```

    **Applications:**
    - 💬 Chatbots sur vos données
    - 📚 Q&A sur documentation
    - 📧 Assistants customer support
    - 🧑‍💼 Analyse de documents légaux/contractuels
    """)

    st.success("""
    🎓 **On verra les RAG en détail dans un prochain cours!**

    Mais vous avez maintenant **toutes les bases** pour:
    - Comprendre comment ça marche
    - Implémenter votre propre système de recherche sémantique
    - L'intégrer dans votre projet web
    - Calculer les coûts et choisir la bonne solution
    """)

    st.balloons()

    st.success("""
    🚀 **Vous avez maintenant toutes les cartes en main!**

    N'hésitez pas à expérimenter et à nous poser des questions pendant le développement de votre projet! 💪
    """)
