"""
Hybrid Search: Combine BM25 (lexical) et Embeddings (sémantique)
"""

from typing import List, Dict
import numpy as np


class HybridSearch:
    """
    Recherche hybride combinant BM25 et Embeddings
    Permet de bénéficier des deux approches: lexicale et sémantique
    """

    def __init__(
        self,
        documents: List[str],
        bm25_engine,
        embedding_engine,
        alpha: float = 0.5
    ):
        """
        Args:
            documents: Liste de documents
            bm25_engine: Instance de BM25Engine
            embedding_engine: Instance de EmbeddingSearch
            alpha: Poids BM25 (0=embeddings only, 1=bm25 only, 0.5=équilibré)
        """
        self.documents = documents
        self.bm25 = bm25_engine
        self.embeddings = embedding_engine
        self.alpha = alpha

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = None
    ) -> List[Dict]:
        """
        Recherche hybride avec combinaison des scores

        Args:
            query: Requête texte
            top_k: Nombre de résultats
            alpha: Override du alpha (optionnel)

        Returns:
            List of {
                'index': int,
                'combined_score': float,
                'bm25_score': float,
                'bm25_score_norm': float,
                'emb_score': float,
                'emb_score_norm': float,
                'document': str
            }
        """
        if alpha is None:
            alpha = self.alpha

        # Scores BM25 (tous les documents)
        bm25_results = self.bm25.search(query, top_k=len(self.documents))
        bm25_scores = {idx: score for idx, score in bm25_results}

        # Scores Embeddings (tous les documents)
        emb_results = self.embeddings.search(query, top_k=len(self.documents))
        emb_scores = {r['index']: r['score'] for r in emb_results}

        # Normalisation min-max
        bm25_norm = self._normalize(bm25_scores)
        emb_norm = self._normalize(emb_scores)

        # Combinaison linéaire
        combined = {}
        for idx in range(len(self.documents)):
            bm25_s = bm25_norm.get(idx, 0)
            emb_s = emb_norm.get(idx, 0)

            combined[idx] = alpha * bm25_s + (1 - alpha) * emb_s

        # Top-k
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                'index': idx,
                'combined_score': score,
                'bm25_score': bm25_scores.get(idx, 0),
                'bm25_score_norm': bm25_norm.get(idx, 0),
                'emb_score': emb_scores.get(idx, 0),
                'emb_score_norm': emb_norm.get(idx, 0),
                'document': self.documents[idx]
            }
            for idx, score in sorted_results
        ]

    def compute_score(
        self,
        query: str,
        doc_index: int,
        alpha: float = None
    ) -> float:
        """
        Calcule le score hybrid pour un document spécifique

        Args:
            query: Requête
            doc_index: Index du document
            alpha: Poids BM25 (optionnel)

        Returns:
            Score combiné normalisé
        """
        if alpha is None:
            alpha = self.alpha

        # Score BM25
        bm25_results = self.bm25.search(query, top_k=len(self.documents))
        bm25_scores = {idx: score for idx, score in bm25_results}

        # Score Embeddings
        emb_results = self.embeddings.search(query, top_k=len(self.documents))
        emb_scores = {r['index']: r['score'] for r in emb_results}

        # Normaliser
        bm25_norm = self._normalize(bm25_scores)
        emb_norm = self._normalize(emb_scores)

        # Combiner
        bm25_s = bm25_norm.get(doc_index, 0)
        emb_s = emb_norm.get(doc_index, 0)

        return alpha * bm25_s + (1 - alpha) * emb_s

    def _normalize(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Normalisation min-max des scores

        Args:
            scores: Dict {doc_index: score}

        Returns:
            Dict {doc_index: normalized_score}
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_val, max_val = min(values), max(values)

        # Si tous les scores sont identiques
        if max_val == min_val:
            return {k: 1.0 for k in scores}

        # Min-max normalization
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def set_alpha(self, alpha: float):
        """Change le paramètre alpha"""
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha

