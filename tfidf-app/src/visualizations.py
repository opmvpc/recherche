"""
Toutes les fonctions de visualisation pour l'application TF-IDF Explorer
Utilise Matplotlib, Seaborn et Plotly pour des graphiques interactifs et pédagogiques
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from wordcloud import WordCloud


# Palette de couleurs cohérente
PRIMARY_COLOR = "#1f77b4"      # Bleu
SECONDARY_COLOR = "#ff7f0e"    # Orange
SUCCESS_COLOR = "#2ca02c"      # Vert
WARNING_COLOR = "#d62728"      # Rouge
NEUTRAL_COLOR = "#7f7f7f"      # Gris
HEATMAP_COLORSCALE = "YlOrRd"  # Jaune → Orange → Rouge


def plot_tf_comparison(documents: List[List[str]], doc_indices: List[int],
                       titles: List[str]) -> plt.Figure:
    """
    Bar chart comparant les TF de plusieurs documents

    Args:
        documents: Liste de documents tokenizés
        doc_indices: Indices des documents à comparer
        titles: Titres des documents

    Returns:
        Figure matplotlib
    """
    from collections import Counter

    fig, axes = plt.subplots(1, len(doc_indices), figsize=(15, 5))

    if len(doc_indices) == 1:
        axes = [axes]

    for idx, (doc_idx, ax) in enumerate(zip(doc_indices, axes)):
        doc = documents[doc_idx]
        counts = Counter(doc)
        total = len(doc)

        # TF normalisé
        tf_dict = {word: count / total for word, count in counts.most_common(10)}

        words = list(tf_dict.keys())
        values = list(tf_dict.values())

        ax.barh(words, values, color=PRIMARY_COLOR)
        ax.set_xlabel('TF (fréquence normalisée)')
        ax.set_title(f'{titles[idx]}\n({total} mots)', fontsize=10)
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_idf_curve(idf_vector: np.ndarray, vocabulary: List[str],
                   documents: List[List[str]]) -> plt.Figure:
    """
    Courbe IDF en fonction du nombre de documents contenant chaque mot

    Args:
        idf_vector: Vecteur IDF
        vocabulary: Liste des mots
        documents: Documents tokenizés

    Returns:
        Figure matplotlib
    """
    # Calculer le nombre de documents contenant chaque mot
    n_docs = len(documents)
    doc_frequencies = []

    for word in vocabulary:
        doc_count = sum(1 for doc in documents if word in doc)
        doc_frequencies.append(doc_count)

    # Créer le dataframe pour le plot
    df = pd.DataFrame({
        'doc_frequency': doc_frequencies,
        'idf': idf_vector
    })

    # Grouper par fréquence documentaire
    df_grouped = df.groupby('doc_frequency')['idf'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_grouped['doc_frequency'], df_grouped['idf'],
            marker='o', linewidth=2, markersize=8, color=PRIMARY_COLOR)

    ax.set_xlabel('Nombre de documents contenant le mot', fontsize=12)
    ax.set_ylabel('IDF (Inverse Document Frequency)', fontsize=12)
    ax.set_title('Impact de la fréquence documentaire sur l\'IDF', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.axhline(y=df_grouped['idf'].mean(), color=WARNING_COLOR,
               linestyle='--', alpha=0.5, label=f'IDF moyen: {df_grouped["idf"].mean():.2f}')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_idf_wordcloud(idf_dict: Dict[str, float], max_words: int = 200) -> plt.Figure:
    """
    Word cloud où la taille des mots est proportionnelle à leur IDF
    ⚡ Optimisé pour grands vocabulaires (limite au top N mots)

    Args:
        idf_dict: Dictionnaire {mot: IDF}
        max_words: Nombre maximum de mots à afficher (défaut 200)

    Returns:
        Figure matplotlib
    """
    # 🚀 OPTIMISATION: Limiter aux top N mots par IDF pour performance
    if len(idf_dict) > max_words:
        sorted_words = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
        idf_dict = dict(sorted_words[:max_words])

    # Créer le word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='YlOrRd',
        relative_scaling=0.5,
        min_font_size=10,
        max_words=max_words  # Limite explicite
    ).generate_from_frequencies(idf_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud IDF (Top {min(len(idf_dict), max_words)} mots - Taille = Importance)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def plot_tfidf_heatmap(tfidf_matrix: np.ndarray, vocabulary: List[str],
                       doc_titles: List[str], top_words: int = 20) -> plt.Figure:
    """
    Heatmap de la matrice TF-IDF (docs × mots)

    Args:
        tfidf_matrix: Matrice TF-IDF
        vocabulary: Liste des mots
        doc_titles: Titres des documents
        top_words: Nombre de mots à afficher

    Returns:
        Figure matplotlib
    """
    # Sélectionner les top mots par variance (les plus discriminants)
    word_variances = np.var(tfidf_matrix, axis=0)
    top_indices = np.argsort(word_variances)[-top_words:]

    top_words_list = [vocabulary[i] for i in top_indices]
    tfidf_subset = tfidf_matrix[:, top_indices]

    # Limiter le nombre de documents affichés si trop grand
    max_docs = 30
    if len(doc_titles) > max_docs:
        # Prendre un échantillon
        doc_indices = np.linspace(0, len(doc_titles)-1, max_docs, dtype=int)
        tfidf_subset = tfidf_subset[doc_indices]
        doc_titles = [doc_titles[i] for i in doc_indices]

    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(doc_titles) * 0.3)))

    sns.heatmap(
        tfidf_subset,
        xticklabels=top_words_list,
        yticklabels=[title[:40] for title in doc_titles],
        cmap=HEATMAP_COLORSCALE,
        cbar_kws={'label': 'Score TF-IDF'},
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel('Mots', fontsize=12, fontweight='bold')
    ax.set_ylabel('Documents', fontsize=12, fontweight='bold')
    ax.set_title(f'Heatmap TF-IDF (Top {top_words} mots discriminants)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_top_words_per_doc(tfidf_matrix: np.ndarray, vocabulary: List[str],
                           doc_index: int, doc_title: str, top_k: int = 10) -> plt.Figure:
    """
    Bar chart des mots les plus importants pour un document

    Args:
        tfidf_matrix: Matrice TF-IDF
        vocabulary: Liste des mots
        doc_index: Index du document
        doc_title: Titre du document
        top_k: Nombre de mots à afficher

    Returns:
        Figure matplotlib
    """
    doc_scores = tfidf_matrix[doc_index]
    top_indices = np.argsort(doc_scores)[-top_k:][::-1]

    top_words = [vocabulary[i] for i in top_indices]
    top_scores = [doc_scores[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(top_words)), top_scores, color=PRIMARY_COLOR)
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words)
    ax.invert_yaxis()

    ax.set_xlabel('Score TF-IDF', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} mots pour: {doc_title}',
                 fontsize=14, fontweight='bold')

    # Ajouter les valeurs sur les barres
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(score, i, f' {score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                            doc_titles: List[str]) -> plt.Figure:
    """
    Heatmap de similarité entre tous les documents

    Args:
        similarity_matrix: Matrice de similarité cosinus
        doc_titles: Titres des documents

    Returns:
        Figure matplotlib
    """
    # Limiter si trop de documents
    max_docs = 50
    if len(doc_titles) > max_docs:
        doc_indices = np.linspace(0, len(doc_titles)-1, max_docs, dtype=int)
        similarity_matrix = similarity_matrix[np.ix_(doc_indices, doc_indices)]
        doc_titles = [doc_titles[i] for i in doc_indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        similarity_matrix,
        xticklabels=[title[:30] for title in doc_titles],
        yticklabels=[title[:30] for title in doc_titles],
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Similarité Cosinus'},
        ax=ax,
        square=True
    )

    ax.set_title('Matrice de Similarité entre Documents',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    return fig


def plot_search_results(results: List[Tuple[int, float]],
                       doc_titles: List[str], query: str) -> plt.Figure:
    """
    Bar chart des résultats de recherche avec scores de similarité

    Args:
        results: Liste de (doc_index, score)
        doc_titles: Titres des documents
        query: Requête de recherche

    Returns:
        Figure matplotlib
    """
    if len(results) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Aucun résultat trouvé',
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

    indices = [r[0] for r in results]
    scores = [r[1] for r in results]
    titles = [doc_titles[i][:50] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5)))

    # Gradient de couleurs
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(scores)))

    bars = ax.barh(range(len(titles)), scores, color=colors)
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Score de Similarité', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_title(f'Résultats de recherche pour: "{query}"',
                 fontsize=14, fontweight='bold')

    # Ajouter les scores sur les barres
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.02, i, f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_documents_3d(tfidf_matrix: np.ndarray, doc_titles: List[str],
                     categories: List[str] = None) -> go.Figure:
    """
    Projection 3D des documents avec PCA (interactif avec Plotly)

    Args:
        tfidf_matrix: Matrice TF-IDF
        doc_titles: Titres des documents
        categories: Catégories des documents (optionnel)

    Returns:
        Figure Plotly
    """
    # Réduction de dimensionnalité avec PCA
    n_components = min(3, tfidf_matrix.shape[1])
    pca = PCA(n_components=n_components)

    coords = pca.fit_transform(tfidf_matrix)

    # Padding si moins de 3 dimensions
    if n_components < 3:
        padding = np.zeros((coords.shape[0], 3 - n_components))
        coords = np.hstack([coords, padding])

    # Créer le scatter 3D
    if categories:
        # Colorier par catégorie
        fig = px.scatter_3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            color=categories,
            hover_name=doc_titles,
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})' if n_components > 1 else 'PC2',
                   'z': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})' if n_components > 2 else 'PC3'},
            title='Projection 3D des Documents (PCA)'
        )
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=8, color=coords[:, 0], colorscale='Viridis'),
            text=doc_titles,
            hoverinfo='text'
        )])

        fig.update_layout(
            title='Projection 3D des Documents (PCA)',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})' if n_components > 1 else 'PC2',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})' if n_components > 2 else 'PC3'
            )
        )

    fig.update_traces(marker=dict(size=6))

    return fig


def plot_documents_2d(tfidf_matrix: np.ndarray, doc_titles: List[str],
                     categories: List[str] = None) -> go.Figure:
    """
    Projection 2D des documents avec PCA (interactif avec Plotly)

    Args:
        tfidf_matrix: Matrice TF-IDF
        doc_titles: Titres des documents
        categories: Catégories des documents (optionnel)

    Returns:
        Figure Plotly
    """
    # Réduction de dimensionnalité avec PCA
    n_components = min(2, tfidf_matrix.shape[1])
    pca = PCA(n_components=n_components)

    coords = pca.fit_transform(tfidf_matrix)

    # Créer le scatter 2D
    if categories:
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1] if n_components > 1 else np.zeros(len(coords)),
            color=categories,
            hover_name=doc_titles,
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})' if n_components > 1 else 'PC2'},
            title='Projection 2D des Documents (PCA)'
        )
    else:
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1] if n_components > 1 else np.zeros(len(coords)),
            hover_name=doc_titles,
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})' if n_components > 1 else 'PC2'},
            title='Projection 2D des Documents (PCA)'
        )

    fig.update_traces(marker=dict(size=10))

    return fig


def plot_tf_vs_tfidf_comparison(tf_scores: Dict[str, float],
                                tfidf_scores: Dict[str, float],
                                top_k: int = 10) -> plt.Figure:
    """
    Comparaison côte à côte des scores TF vs TF-IDF

    Args:
        tf_scores: Scores TF {mot: score}
        tfidf_scores: Scores TF-IDF {mot: score}
        top_k: Nombre de mots à afficher

    Returns:
        Figure matplotlib
    """
    # Top mots pour chaque métrique
    top_tf = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # TF
    words_tf = [w for w, s in top_tf]
    scores_tf = [s for w, s in top_tf]
    ax1.barh(range(len(words_tf)), scores_tf, color=SECONDARY_COLOR)
    ax1.set_yticks(range(len(words_tf)))
    ax1.set_yticklabels(words_tf)
    ax1.invert_yaxis()
    ax1.set_xlabel('Score TF', fontweight='bold')
    ax1.set_title('Term Frequency (TF)', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    # TF-IDF
    words_tfidf = [w for w, s in top_tfidf]
    scores_tfidf = [s for w, s in top_tfidf]
    ax2.barh(range(len(words_tfidf)), scores_tfidf, color=PRIMARY_COLOR)
    ax2.set_yticks(range(len(words_tfidf)))
    ax2.set_yticklabels(words_tfidf)
    ax2.invert_yaxis()
    ax2.set_xlabel('Score TF-IDF', fontweight='bold')
    ax2.set_title('TF-IDF (avec IDF)', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle('Comparaison TF vs TF-IDF', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_saturation_effect(
    k1_values: List[float] = [0.5, 1.2, 1.5, 2.0],
    max_freq: int = 50
) -> plt.Figure:
    """
    Graphique montrant l'effet de saturation pour différents k1
    Compare BM25 (avec saturation) vs TF-IDF (linéaire)

    Args:
        k1_values: Liste des valeurs k1 à visualiser
        max_freq: Fréquence maximale sur l'axe X

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    frequencies = list(range(1, max_freq + 1))

    # TF-IDF linéaire (pour comparaison)
    tfidf_scores = frequencies  # Linéaire, pas de saturation
    ax.plot(frequencies, tfidf_scores,
            label='TF-IDF (linéaire)',
            linestyle='--',
            linewidth=3,
            color='#d62728',
            marker='')

    # BM25 pour différents k1
    colors = ['#17becf', '#1f77b4', '#2ca02c', '#ff7f0e']

    for k1, color in zip(k1_values, colors):
        bm25_scores = []
        for f in frequencies:
            score = f * (k1 + 1) / (f + k1)
            bm25_scores.append(score)

        label = f'BM25 k1={k1}'
        if k1 == 1.5:
            label += ' ⭐ (standard)'

        ax.plot(frequencies, bm25_scores,
                label=label,
                linewidth=2.5,
                color=color,
                marker='o' if k1 == 1.5 else '',
                markersize=4)

    ax.set_xlabel('Nombre d\'occurrences du mot', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (normalisé)', fontsize=13, fontweight='bold')
    ax.set_title('Effet de Saturation: BM25 vs TF-IDF', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate('TF-IDF continue de croître\nlinéairement (problème!)',
                xy=(40, 40), xytext=(30, 55),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')

    ax.annotate('BM25 atteint un plateau\n(saturation intelligente)',
                xy=(40, 2.3), xytext=(25, 10),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_length_normalization(
    avgdl: float = 100,
    doc_lengths: List[int] = [50, 100, 150, 200],
    b_values: List[float] = [0.0, 0.5, 0.75, 1.0]
) -> plt.Figure:
    """
    Heatmap ou grouped bar chart montrant l'effet de b sur la normalisation

    Args:
        avgdl: Longueur moyenne du corpus
        doc_lengths: Longueurs de documents à tester
        b_values: Valeurs de b à visualiser

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(doc_lengths))
    width = 0.2

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    for i, (b, color) in enumerate(zip(b_values, colors)):
        norms = []
        for length in doc_lengths:
            norm = 1 - b + b * (length / avgdl)
            norms.append(norm)

        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, norms, width, label=f'b={b}', color=color, alpha=0.8)

        # Ajouter les valeurs sur les barres
        for bar, norm in zip(bars, norms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{norm:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Longueur du document (mots)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Facteur de normalisation', fontsize=13, fontweight='bold')
    ax.set_title(f'Impact de b sur la Normalisation de Longueur (avgdl={avgdl})',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l} mots' for l in doc_lengths])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Pas de pénalité')

    # Annotations
    ax.annotate('Facteur > 1 = pénalité\n(docs longs)',
                xy=(2.5, 1.5), xytext=(3, 1.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')

    ax.annotate('Facteur < 1 = boost\n(docs courts)',
                xy=(0.5, 0.5), xytext=(0.8, 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_parameter_space_heatmap(
    bm25_engine,
    query: str,
    doc_index: int,
    k1_range: Tuple[float, float] = (0.5, 3.0),
    b_range: Tuple[float, float] = (0.0, 1.0),
    resolution: int = 20
) -> go.Figure:
    """
    Heatmap 2D interactive: k1 vs b, couleur = score BM25
    Permet de visualiser l'espace des paramètres

    Args:
        bm25_engine: Instance de BM25Engine
        query: Requête de test
        doc_index: Index du document à scorer
        k1_range: (min, max) pour k1
        b_range: (min, max) pour b
        resolution: Nombre de points par dimension

    Returns:
        Figure Plotly interactive
    """
    k1_values = np.linspace(k1_range[0], k1_range[1], resolution)
    b_values = np.linspace(b_range[0], b_range[1], resolution)

    scores = np.zeros((resolution, resolution))

    # Sauvegarder les paramètres originaux
    orig_k1, orig_b = bm25_engine.k1, bm25_engine.b

    # Tester toutes les combinaisons
    for i, k1 in enumerate(k1_values):
        for j, b in enumerate(b_values):
            bm25_engine.update_parameters(k1=k1, b=b)
            results = bm25_engine.search(query, top_k=1)

            # Trouver le score pour notre doc spécifique
            score = 0.0
            for idx, s in results:
                if idx == doc_index:
                    score = s
                    break

            scores[j, i] = score

    # Restaurer les paramètres
    bm25_engine.update_parameters(k1=orig_k1, b=orig_b)

    # Créer la heatmap
    fig = go.Figure(data=go.Heatmap(
        z=scores,
        x=k1_values,
        y=b_values,
        colorscale='YlOrRd',
        hovertemplate='k1: %{x:.2f}<br>b: %{y:.2f}<br>Score: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Score BM25')
    ))

    # Marquer le point standard (k1=1.5, b=0.75)
    fig.add_trace(go.Scatter(
        x=[1.5],
        y=[0.75],
        mode='markers+text',
        marker=dict(size=15, color='white', symbol='star', line=dict(color='black', width=2)),
        text=['Standard'],
        textposition='top center',
        textfont=dict(size=12, color='white'),
        name='Paramètres standard',
        showlegend=True
    ))

    fig.update_layout(
        title='Espace des Paramètres BM25 (k1 vs b)',
        xaxis_title='k1 (saturation)',
        yaxis_title='b (normalisation)',
        width=800,
        height=600,
        font=dict(size=12)
    )

    return fig


def plot_tfidf_bm25_comparison(
    tfidf_scores: List[Tuple[int, float]],
    bm25_scores: List[Tuple[int, float]],
    doc_titles: List[str],
    query: str,
    top_k: int = 10
) -> plt.Figure:
    """
    Grouped bar chart comparant les scores TF-IDF vs BM25

    Args:
        tfidf_scores: Résultats TF-IDF [(idx, score), ...]
        bm25_scores: Résultats BM25 [(idx, score), ...]
        doc_titles: Titres des documents
        query: Requête recherchée
        top_k: Nombre de résultats à afficher

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extraire top k pour chaque
    tfidf_top = tfidf_scores[:top_k]
    bm25_top = bm25_scores[:top_k]

    # Créer l'union des documents (certains peuvent être différents)
    all_indices = list(set([idx for idx, _ in tfidf_top] + [idx for idx, _ in bm25_top]))
    all_indices = sorted(all_indices, key=lambda idx: -max(
        next((s for i, s in bm25_top if i == idx), 0),
        next((s for i, s in tfidf_top if i == idx), 0)
    ))[:top_k]

    # Préparer les données
    labels = [doc_titles[idx][:40] + '...' if len(doc_titles[idx]) > 40 else doc_titles[idx]
              for idx in all_indices]

    tfidf_vals = [next((s for i, s in tfidf_top if i == idx), 0) for idx in all_indices]
    bm25_vals = [next((s for i, s in bm25_top if i == idx), 0) for idx in all_indices]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.barh(x - width/2, tfidf_vals, width, label='TF-IDF', color='#d62728', alpha=0.8)
    bars2 = ax.barh(x + width/2, bm25_vals, width, label='BM25', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('Documents', fontsize=12, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparaison TF-IDF vs BM25 pour: "{query}"', fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    # Ajouter les valeurs
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, tfidf_vals, bm25_vals)):
        if val1 > 0:
            ax.text(val1, i - width/2, f' {val1:.3f}', va='center', fontsize=8)
        if val2 > 0:
            ax.text(val2, i + width/2, f' {val2:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    return fig


def plot_score_distributions(
    tfidf_scores: List[float],
    bm25_scores: List[float]
) -> plt.Figure:
    """
    Histogrammes overlaid montrant la distribution des scores
    BM25 devrait avoir une meilleure séparation

    Args:
        tfidf_scores: Liste de tous les scores TF-IDF
        bm25_scores: Liste de tous les scores BM25

    Returns:
        Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # TF-IDF distribution
    ax1.hist(tfidf_scores, bins=30, color='#d62728', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(tfidf_scores), color='blue', linestyle='--', linewidth=2,
                label=f'Moyenne: {np.mean(tfidf_scores):.3f}')
    ax1.axvline(np.median(tfidf_scores), color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {np.median(tfidf_scores):.3f}')
    ax1.set_xlabel('Score TF-IDF', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution TF-IDF', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # BM25 distribution
    ax2.hist(bm25_scores, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(bm25_scores), color='blue', linestyle='--', linewidth=2,
                label=f'Moyenne: {np.mean(bm25_scores):.3f}')
    ax2.axvline(np.median(bm25_scores), color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {np.median(bm25_scores):.3f}')
    ax2.set_xlabel('Score BM25', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution BM25', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparaison des Distributions de Scores', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_vocabulary_stats(documents: List[List[str]]) -> plt.Figure:
    """
    Statistiques sur le vocabulaire et la longueur des documents

    Args:
        documents: Documents tokenizés

    Returns:
        Figure matplotlib
    """
    # Calculs
    doc_lengths = [len(doc) for doc in documents]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution de longueur
    axes[0].hist(doc_lengths, bins=30, color=PRIMARY_COLOR, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(doc_lengths), color=WARNING_COLOR,
                   linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(doc_lengths):.1f}')
    axes[0].set_xlabel('Nombre de mots', fontweight='bold')
    axes[0].set_ylabel('Nombre de documents', fontweight='bold')
    axes[0].set_title('Distribution de la longueur des documents', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Top mots les plus fréquents dans le corpus
    from collections import Counter
    all_words = [word for doc in documents for word in doc]
    word_counts = Counter(all_words).most_common(15)

    words = [w for w, c in word_counts]
    counts = [c for w, c in word_counts]

    axes[1].barh(range(len(words)), counts, color=SUCCESS_COLOR)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Nombre d\'occurrences', fontweight='bold')
    axes[1].set_title('Mots les plus fréquents du corpus', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# VISUALISATIONS POUR EMBEDDINGS
# ============================================================================

def plot_embedding_space_3d(
    embeddings: np.ndarray,
    labels: List[str],
    categories: List[str] = None,
    query_embedding: np.ndarray = None,
    query_label: str = "Query",
    top_k_indices: List[int] = None,
    max_points: int = 1000
) -> go.Figure:
    """
    Visualisation 3D interactive de l'espace vectoriel avec PCA
    ⚡ Optimisé: Échantillonne automatiquement si trop de points

    Args:
        embeddings: Embeddings (n_docs × n_dims)
        labels: Titres des documents
        categories: Catégories des documents (optionnel)
        query_embedding: Embedding de la query (optionnel)
        query_label: Label de la query
        top_k_indices: Indices des top résultats (optionnel)
        max_points: Nombre max de points à afficher (défaut 1000)

    Returns:
        Figure Plotly interactive
    """
    from sklearn.decomposition import PCA
    import random

    # 🚀 OPTIMISATION: Échantillonnage si trop de documents
    n_docs = len(embeddings)
    sampled_indices = None

    if n_docs > max_points:
        # Toujours inclure les top_k si fournis
        if top_k_indices:
            must_include = set(top_k_indices[:20])  # Top 20 minimum
            remaining_slots = max_points - len(must_include)
            other_indices = [i for i in range(n_docs) if i not in must_include]
            sampled_indices = list(must_include) + random.sample(other_indices, min(remaining_slots, len(other_indices)))
        else:
            sampled_indices = random.sample(range(n_docs), max_points)

        sampled_indices = sorted(sampled_indices)
        embeddings = embeddings[sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        if categories:
            categories = [categories[i] for i in sampled_indices]

        # Réajuster top_k_indices si nécessaire
        if top_k_indices:
            idx_map = {old: new for new, old in enumerate(sampled_indices)}
            top_k_indices = [idx_map[i] for i in top_k_indices if i in idx_map]

    # Réduction 3D avec PCA
    pca = PCA(n_components=3)

    if query_embedding is not None:
        # Inclure la query dans la réduction
        all_embeddings = np.vstack([embeddings, query_embedding.reshape(1, -1)])
        reduced = pca.fit_transform(all_embeddings)
        docs_reduced = reduced[:-1]
        query_reduced = reduced[-1]
    else:
        docs_reduced = pca.fit_transform(embeddings)
        query_reduced = None

    # Créer la figure
    fig = go.Figure()

    # Documents
    if categories:
        unique_cats = list(set(categories))
        colors_map = px.colors.qualitative.Plotly
        for i, cat in enumerate(unique_cats):
            mask = np.array([c == cat for c in categories])
            indices = np.where(mask)[0]

            fig.add_trace(go.Scatter3d(
                x=docs_reduced[mask, 0],
                y=docs_reduced[mask, 1],
                z=docs_reduced[mask, 2],
                mode='markers',
                name=cat,
                marker=dict(
                    size=6,
                    color=colors_map[i % len(colors_map)],
                    opacity=0.7
                ),
                text=[labels[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>Catégorie: ' + cat + '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=docs_reduced[:, 0],
            y=docs_reduced[:, 1],
            z=docs_reduced[:, 2],
            mode='markers',
            name='Documents',
            marker=dict(
                size=6,
                color=PRIMARY_COLOR,
                opacity=0.7
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

    # Query
    if query_reduced is not None:
        fig.add_trace(go.Scatter3d(
            x=[query_reduced[0]],
            y=[query_reduced[1]],
            z=[query_reduced[2]],
            mode='markers',
            name=query_label,
            marker=dict(
                size=15,
                color=WARNING_COLOR,
                symbol='diamond',
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>' + query_label + '</b><extra></extra>'
        ))

        # Lignes vers top-k résultats
        if top_k_indices:
            for idx in top_k_indices[:3]:  # Top 3
                fig.add_trace(go.Scatter3d(
                    x=[query_reduced[0], docs_reduced[idx, 0]],
                    y=[query_reduced[1], docs_reduced[idx, 1]],
                    z=[query_reduced[2], docs_reduced[idx, 2]],
                    mode='lines',
                    line=dict(color=SUCCESS_COLOR, width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Titre avec indication d'échantillonnage si nécessaire
    title_text = 'Espace Vectoriel 3D (PCA)'
    if sampled_indices is not None:
        title_text += f' - {len(embeddings)}/{n_docs} docs échantillonnés'

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
        ),
        hovermode='closest',
        width=900,
        height=700
    )

    return fig


def plot_tsne_2d(
    embeddings: np.ndarray,
    labels: List[str],
    categories: List[str] = None,
    query_embedding: np.ndarray = None,
    query_label: str = "Query",
    perplexity: int = 30,
    max_points: int = 500
) -> go.Figure:
    """
    Visualisation 2D avec t-SNE pour clustering
    ⚡ Optimisé: t-SNE est LENT! Limite à 500 docs max par défaut

    Args:
        embeddings: Embeddings (n_docs × n_dims)
        labels: Titres des documents
        categories: Catégories (optionnel)
        query_embedding: Embedding query (optionnel)
        query_label: Label query
        perplexity: Paramètre t-SNE
        max_points: Nombre max de points (défaut 500, t-SNE est O(n²)!)

    Returns:
        Figure Plotly
    """
    from sklearn.manifold import TSNE
    import random

    # 🚀 OPTIMISATION CRITIQUE: t-SNE est O(n²)! Limiter à 500 docs
    n_docs = len(embeddings)
    sampled_indices = None

    if n_docs > max_points:
        sampled_indices = random.sample(range(n_docs), max_points)
        sampled_indices = sorted(sampled_indices)
        embeddings = embeddings[sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        if categories:
            categories = [categories[i] for i in sampled_indices]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1), random_state=42)

    if query_embedding is not None:
        all_embeddings = np.vstack([embeddings, query_embedding.reshape(1, -1)])
        reduced = tsne.fit_transform(all_embeddings)
        docs_reduced = reduced[:-1]
        query_reduced = reduced[-1]
    else:
        docs_reduced = tsne.fit_transform(embeddings)
        query_reduced = None

    # Figure
    fig = go.Figure()

    # Documents
    if categories:
        unique_cats = list(set(categories))
        colors_map = px.colors.qualitative.Safe
        for i, cat in enumerate(unique_cats):
            mask = np.array([c == cat for c in categories])
            indices = np.where(mask)[0]

            fig.add_trace(go.Scatter(
                x=docs_reduced[mask, 0],
                y=docs_reduced[mask, 1],
                mode='markers',
                name=cat,
                marker=dict(
                    size=10,
                    color=colors_map[i % len(colors_map)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[labels[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>Catégorie: ' + cat + '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=docs_reduced[:, 0],
            y=docs_reduced[:, 1],
            mode='markers',
            name='Documents',
            marker=dict(
                size=10,
                color=PRIMARY_COLOR,
                opacity=0.7
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

    # Query
    if query_reduced is not None:
        fig.add_trace(go.Scatter(
            x=[query_reduced[0]],
            y=[query_reduced[1]],
            mode='markers',
            name=query_label,
            marker=dict(
                size=20,
                color=WARNING_COLOR,
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>' + query_label + '</b><extra></extra>'
        ))

    # Titre avec warning si échantillonné
    title_text = 'Visualisation 2D (t-SNE) - Clustering Sémantique'
    if sampled_indices is not None:
        title_text += f' ⚡ {len(embeddings)}/{n_docs} docs échantillonnés'

    fig.update_layout(
        title=title_text,
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        hovermode='closest',
        width=900,
        height=700,
        showlegend=True
    )

    return fig


def plot_similarity_heatmap_embeddings(
    similarity_matrix: np.ndarray,
    labels: List[str],
    top_n: int = 20
) -> plt.Figure:
    """
    Heatmap de similarité entre documents (embeddings)

    Args:
        similarity_matrix: Matrice de similarité (n × n)
        labels: Labels des documents
        top_n: Nombre de documents à afficher

    Returns:
        Figure matplotlib
    """
    # Sélectionner les top_n premiers
    sim_subset = similarity_matrix[:top_n, :top_n]
    labels_subset = labels[:top_n]

    # Tronquer les labels si trop longs
    labels_short = [l[:30] + '...' if len(l) > 30 else l for l in labels_subset]

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        sim_subset,
        annot=False,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Similarité Cosinus'},
        xticklabels=labels_short,
        yticklabels=labels_short,
        ax=ax
    )

    plt.title('Matrice de Similarité Sémantique (Embeddings)', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Documents', fontweight='bold', fontsize=12)
    plt.ylabel('Documents', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    return fig


def plot_clustering_2d(
    embeddings: np.ndarray,
    labels: List[str],
    n_clusters: int = 3,
    max_points: int = 500
) -> plt.Figure:
    """
    Clustering K-means avec visualisation t-SNE
    ⚡ Optimisé: Échantillonnage automatique si > 500 docs

    Args:
        embeddings: Embeddings (n × dims)
        labels: Labels documents
        n_clusters: Nombre de clusters
        max_points: Nombre max de points (défaut 500)

    Returns:
        Figure matplotlib
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import random

    # 🚀 OPTIMISATION: t-SNE est O(n²)! Échantillonner si nécessaire
    n_docs = len(embeddings)
    sampled_indices = None

    if n_docs > max_points:
        sampled_indices = random.sample(range(n_docs), max_points)
        sampled_indices = sorted(sampled_indices)
        embeddings = embeddings[sampled_indices]
        labels = [labels[i] for i in sampled_indices]

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # t-SNE pour visualisation
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=clusters,
        cmap='tab10',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Annotations pour quelques documents
    step = max(1, len(labels) // 15)
    for i in range(0, len(labels), step):
        ax.annotate(
            labels[i][:20] + '...' if len(labels[i]) > 20 else labels[i],
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    plt.colorbar(scatter, label='Cluster ID', ax=ax)

    # Titre avec indication d'échantillonnage
    title = 'Clustering Automatique des Documents (K-means + t-SNE)'
    if sampled_indices is not None:
        title += f'\n⚡ {len(embeddings)}/{n_docs} docs échantillonnés'

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_technique_comparison_radar(
    metrics: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Graphique radar comparant TF-IDF/BM25/Embeddings/Hybrid

    Args:
        metrics: Dict {technique: {metric: value}}
                Example: {'TF-IDF': {'precision': 0.45, 'recall': 0.52, ...}, ...}

    Returns:
        Figure Plotly radar
    """
    fig = go.Figure()

    # Extraire les métriques communes
    metric_names = list(next(iter(metrics.values())).keys())

    for technique, values in metrics.items():
        metric_values = [values[m] for m in metric_names]
        metric_values.append(metric_values[0])  # Fermer le polygone

        fig.add_trace(go.Scatterpolar(
            r=metric_values,
            theta=metric_names + [metric_names[0]],
            name=technique,
            fill='toself'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Comparaison des Techniques de Recherche",
        showlegend=True,
        width=800,
        height=600
    )

    return fig


def plot_hybrid_alpha_effect(
    alpha_values: List[float],
    scores: List[float],
    current_alpha: float,
    doc_label: str
) -> plt.Figure:
    """
    Graphique montrant l'effet du paramètre alpha sur le score hybrid

    Args:
        alpha_values: Valeurs de alpha testées
        scores: Scores correspondants
        current_alpha: Alpha actuel
        doc_label: Label du document

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(alpha_values, scores, linewidth=3, marker='o', markersize=8, color=PRIMARY_COLOR)
    ax.axvline(current_alpha, color=WARNING_COLOR, linestyle='--', linewidth=2,
               label=f'α actuel = {current_alpha:.2f}')

    ax.set_xlabel('α (poids BM25)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Final', fontsize=12, fontweight='bold')
    ax.set_title(f'Évolution du Score selon α\nDocument: {doc_label}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Annotations aux extrémités
    ax.text(0.02, ax.get_ylim()[0] + 0.05, '100% Embeddings', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.98, ax.get_ylim()[0] + 0.05, '100% BM25', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), ha='right')

    plt.tight_layout()
    return fig


def plot_multi_technique_comparison(
    results_dict: Dict[str, List[Tuple[int, float]]],
    titles: List[str],
    query: str,
    top_k: int = 10
) -> plt.Figure:
    """
    Comparaison side-by-side de TF-IDF, BM25, Embeddings

    Args:
        results_dict: {'TF-IDF': [(idx, score), ...], 'BM25': [...], 'Embeddings': [...]}
        titles: Titres des documents
        query: Query utilisée
        top_k: Nombre de résultats à afficher

    Returns:
        Figure matplotlib
    """
    techniques = list(results_dict.keys())
    n_techniques = len(techniques)

    fig, axes = plt.subplots(1, n_techniques, figsize=(7 * n_techniques, 8))

    if n_techniques == 1:
        axes = [axes]

    colors_map = {'TF-IDF': '#d62728', 'BM25': '#2ca02c', 'Embeddings': '#1f77b4', 'Hybrid': '#ff7f0e'}

    for ax, technique in zip(axes, techniques):
        results = results_dict[technique][:top_k]

        # Extraire scores et labels
        doc_indices = [idx for idx, _ in results]
        scores = [score for _, score in results]
        labels = [titles[idx][:30] + '...' if len(titles[idx]) > 30 else titles[idx]
                  for idx in doc_indices]

        # Bar chart horizontal
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, scores, color=colors_map.get(technique, PRIMARY_COLOR), edgecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(f'{technique}', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Annotations scores
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width, i, f' {score:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Comparaison Multi-Techniques\nQuery: "{query}"', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig
