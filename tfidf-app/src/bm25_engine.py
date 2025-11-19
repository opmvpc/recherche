"""
Implémentation pédagogique de BM25 (Best Matching 25) from scratch
Améliore TF-IDF avec saturation du TF et normalisation paramétrable
"""

import math
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import re


def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocessing simple du texte (même que TF-IDF pour cohérence)

    Args:
        text: Texte brut
        remove_stopwords: Supprimer les stopwords français

    Returns:
        Liste de tokens
    """
    stopwords = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'cette',
        'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'à', 'au', 'aux',
        'par', 'pour', 'sur', 'dans', 'avec', 'sans', 'sous', 'vers',
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'qui', 'que', 'quoi', 'dont', 'où', 'si', 'plus', 'très',
        'est', 'sont', 'être', 'avoir', 'fait', 'faire', 'ça'
    }

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    return tokens


class BM25Engine:
    """
    Implémentation pédagogique de BM25
    Conserve tous les états intermédiaires pour visualisation et comparaison
    """

    def __init__(
        self,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True
    ):
        """
        Initialise le moteur BM25

        Args:
            documents: Liste de textes bruts
            k1: Paramètre de saturation (1.2-2.0, défaut 1.5)
            b: Paramètre de normalisation de longueur (0.0-1.0, défaut 0.75)
            remove_stopwords: Supprimer les stopwords
        """
        self.raw_documents = documents
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        # Preprocessing
        self.documents = [preprocess_text(doc, remove_stopwords) for doc in documents]

        # Statistiques du corpus
        self.N = len(self.documents)  # Nombre total de documents
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Construction du vocabulaire
        all_words = set()
        for doc in self.documents:
            all_words.update(doc)
        self.vocabulary = sorted(list(all_words))

        # IDF pré-calculé pour chaque mot
        self.idf = self._compute_idf()

        # Document frequencies (nombre de docs contenant chaque mot)
        self.doc_freqs = self._compute_doc_frequencies()

        self.is_fitted = True

    def _compute_doc_frequencies(self) -> Dict[str, int]:
        """
        Calcule le nombre de documents contenant chaque mot

        Returns:
            Dict {mot: nombre_de_documents}
        """
        doc_freqs = {}
        for word in self.vocabulary:
            count = sum(1 for doc in self.documents if word in doc)
            doc_freqs[word] = count
        return doc_freqs

    def _compute_idf(self) -> Dict[str, float]:
        """
        Calcule l'IDF avec le smoothing BM25

        Formule BM25: IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5))
        où N = nombre total de docs, n(q) = nombre de docs contenant q

        Returns:
            Dict {mot: IDF}
        """
        idf = {}

        for word in self.vocabulary:
            # Nombre de documents contenant le mot
            n = sum(1 for doc in self.documents if word in doc)

            # IDF BM25 avec smoothing
            idf_value = math.log((self.N - n + 0.5) / (n + 0.5))

            idf[word] = idf_value

        return idf

    def _score_document(
        self,
        query_tokens: List[str],
        doc_index: int
    ) -> float:
        """
        Calcule le score BM25 d'un document pour une query

        Formule complète:
        BM25 = Σ IDF(q_i) × (f(q_i, D) × (k1 + 1)) / (f(q_i, D) + k1 × (1 - b + b × |D|/avgdl))

        Args:
            query_tokens: Tokens de la query
            doc_index: Index du document à scorer

        Returns:
            Score BM25
        """
        score = 0.0
        doc = self.documents[doc_index]
        doc_len = self.doc_lengths[doc_index]

        # Compter les occurrences dans le document
        doc_freqs = Counter(doc)

        # Facteur de normalisation de longueur
        norm_factor = 1 - self.b + self.b * (doc_len / self.avgdl)

        # Scorer chaque terme de la query
        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            # Fréquence du terme dans le document
            f = doc_freqs.get(term, 0)

            if f == 0:
                continue

            # IDF du terme
            idf = self.idf[term]

            # Formule BM25
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * norm_factor

            term_score = idf * (numerator / denominator)
            score += term_score

        return score

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Recherche les documents les plus pertinents pour une query

        Args:
            query: Texte de la requête
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de (doc_index, score) triée par score décroissant
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté")

        # Preprocessing de la query
        query_tokens = preprocess_text(query, self.remove_stopwords)

        if len(query_tokens) == 0:
            return []

        # Scorer tous les documents
        scores = []
        for doc_idx in range(self.N):
            score = self._score_document(query_tokens, doc_idx)
            scores.append((doc_idx, score))

        # Trier par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def explain(
        self,
        query: str,
        doc_index: int
    ) -> Dict:
        """
        Explication détaillée du calcul BM25 pour un document

        Args:
            query: Texte de la requête
            doc_index: Index du document

        Returns:
            Dict avec tous les calculs intermédiaires
        """
        query_tokens = preprocess_text(query, self.remove_stopwords)
        doc = self.documents[doc_index]
        doc_len = self.doc_lengths[doc_index]
        doc_freqs = Counter(doc)

        # Facteur de normalisation
        norm_factor = 1 - self.b + self.b * (doc_len / self.avgdl)

        # Détails par terme
        term_details = []
        total_score = 0.0

        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            f = doc_freqs.get(term, 0)
            idf = self.idf[term]

            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * norm_factor
            term_score = idf * (numerator / denominator) if denominator > 0 else 0.0

            total_score += term_score

            term_details.append({
                'term': term,
                'frequency': f,
                'idf': idf,
                'numerator': numerator,
                'denominator': denominator,
                'score': term_score
            })

        return {
            'query': query,
            'query_tokens': query_tokens,
            'doc_index': doc_index,
            'doc_length': doc_len,
            'avgdl': self.avgdl,
            'k1': self.k1,
            'b': self.b,
            'norm_factor': norm_factor,
            'term_details': term_details,
            'total_score': total_score
        }

    def update_parameters(
        self,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ):
        """
        Met à jour les paramètres k1 et b sans réindexer

        Args:
            k1: Nouveau paramètre de saturation (None = pas de changement)
            b: Nouveau paramètre de normalisation (None = pas de changement)
        """
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

    def compare_with_tfidf(
        self,
        query: str,
        tfidf_scores: List[Tuple[int, float]],
        top_k: int = 10
    ) -> Dict:
        """
        Compare les résultats BM25 avec TF-IDF

        Args:
            query: Requête de recherche
            tfidf_scores: Résultats TF-IDF [(doc_idx, score), ...]
            top_k: Nombre de résultats à comparer

        Returns:
            Dict avec métriques de comparaison
        """
        bm25_scores = self.search(query, top_k=top_k)

        # Extraire les indices
        bm25_indices = set([idx for idx, _ in bm25_scores])
        tfidf_indices = set([idx for idx, _ in tfidf_scores[:top_k]])

        # Overlap (combien en commun)
        overlap = len(bm25_indices.intersection(tfidf_indices))
        overlap_pct = (overlap / top_k * 100) if top_k > 0 else 0

        # Rank correlation (Spearman simplifiée)
        # Pour chaque doc dans les deux tops, calculer la différence de rang
        rank_diffs = []
        for idx in bm25_indices.intersection(tfidf_indices):
            bm25_rank = next(i for i, (doc_idx, _) in enumerate(bm25_scores) if doc_idx == idx)
            tfidf_rank = next(i for i, (doc_idx, _) in enumerate(tfidf_scores) if doc_idx == idx)
            rank_diffs.append(abs(bm25_rank - tfidf_rank))

        avg_rank_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0

        return {
            'bm25_results': bm25_scores,
            'tfidf_results': tfidf_scores[:top_k],
            'overlap': overlap,
            'overlap_percentage': overlap_pct,
            'avg_rank_difference': avg_rank_diff,
            'bm25_only': bm25_indices - tfidf_indices,
            'tfidf_only': tfidf_indices - bm25_indices
        }

    def get_term_saturation_curve(
        self,
        max_freq: int = 50,
        k1_values: List[float] = None
    ) -> Dict[float, List[float]]:
        """
        Génère les courbes de saturation pour différentes valeurs de k1
        Utile pour visualiser l'effet du paramètre k1

        Args:
            max_freq: Fréquence maximale à calculer
            k1_values: Liste des valeurs k1 à tester (None = valeurs par défaut)

        Returns:
            Dict {k1: [scores pour freq 1..max_freq]}
        """
        if k1_values is None:
            k1_values = [0.5, 1.2, 1.5, 2.0]

        curves = {}

        for k1 in k1_values:
            scores = []
            for f in range(1, max_freq + 1):
                # Formule de saturation: f * (k1 + 1) / (f + k1)
                # (sans IDF ni normalisation pour isoler l'effet)
                score = f * (k1 + 1) / (f + k1)
                scores.append(score)
            curves[k1] = scores

        return curves

    def get_length_normalization_factors(
        self,
        doc_lengths: List[int],
        b_values: List[float] = None
    ) -> Dict[float, List[float]]:
        """
        Calcule les facteurs de normalisation pour différentes longueurs et valeurs de b

        Args:
            doc_lengths: Liste des longueurs de documents à tester
            b_values: Liste des valeurs b à tester (None = valeurs par défaut)

        Returns:
            Dict {b: [facteurs pour chaque longueur]}
        """
        if b_values is None:
            b_values = [0.0, 0.5, 0.75, 1.0]

        factors = {}

        for b in b_values:
            norms = []
            for length in doc_lengths:
                # Formule: 1 - b + b * (length / avgdl)
                factor = 1 - b + b * (length / self.avgdl)
                norms.append(factor)
            factors[b] = norms

        return factors

