"""
Implémentation pédagogique de TF-IDF from scratch
Conserve tous les états intermédiaires pour visualisation
"""

from typing import List, Dict, Tuple
import numpy as np
import re
from collections import Counter


def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocessing simple du texte

    Args:
        text: Texte brut à traiter
        remove_stopwords: Supprimer les stopwords français

    Returns:
        Liste de tokens (mots)
    """
    # Stopwords français basiques
    stopwords = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'cette',
        'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'à', 'au', 'aux',
        'par', 'pour', 'sur', 'dans', 'avec', 'sans', 'sous', 'vers',
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'qui', 'que', 'quoi', 'dont', 'où', 'si', 'plus', 'très',
        'est', 'sont', 'être', 'avoir', 'fait', 'faire', 'ça'
    }

    # Lowercase
    text = text.lower()

    # Suppression ponctuation et caractères spéciaux
    text = re.sub(r'[^\w\s]', ' ', text)

    # Split sur espaces
    tokens = text.split()

    # Suppression stopwords si demandé
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    return tokens


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs

    Args:
        vec1: Premier vecteur
        vec2: Deuxième vecteur

    Returns:
        Similarité cosinus (entre 0 et 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class TFIDFEngine:
    """
    Implémentation pédagogique de TF-IDF
    Conserve tous les états intermédiaires pour visualisation
    """

    def __init__(self, documents: List[str], remove_stopwords: bool = True):
        """
        Initialise le moteur TF-IDF

        Args:
            documents: Liste de textes bruts
            remove_stopwords: Supprimer les stopwords français
        """
        self.raw_documents = documents
        self.remove_stopwords = remove_stopwords

        # Preprocessing
        self.documents = [preprocess_text(doc, remove_stopwords) for doc in documents]

        # États intermédiaires (seront calculés par fit())
        self.vocabulary: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.tf_matrix: np.ndarray = None  # Shape: (n_docs, n_vocab)
        self.idf_vector: np.ndarray = None  # Shape: (n_vocab,)
        self.tfidf_matrix: np.ndarray = None  # Shape: (n_docs, n_vocab)

        self.is_fitted = False

    def fit(self):
        """
        Calcule TF, IDF, et TF-IDF pour tous les documents
        """
        # 1. Construire le vocabulaire
        all_words = set()
        for doc in self.documents:
            all_words.update(doc)

        self.vocabulary = sorted(list(all_words))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        n_docs = len(self.documents)
        n_vocab = len(self.vocabulary)

        # 2. Calculer la matrice TF
        self.tf_matrix = np.zeros((n_docs, n_vocab))

        for doc_idx, doc in enumerate(self.documents):
            tf_dict = self.compute_tf(doc_idx)
            for word, tf_value in tf_dict.items():
                word_idx = self.word_to_idx[word]
                self.tf_matrix[doc_idx, word_idx] = tf_value

        # 3. Calculer le vecteur IDF
        self.idf_vector = np.zeros(n_vocab)
        idf_dict = self.compute_idf()

        for word, idf_value in idf_dict.items():
            word_idx = self.word_to_idx[word]
            self.idf_vector[word_idx] = idf_value

        # 4. Calculer la matrice TF-IDF
        self.compute_tfidf()

        self.is_fitted = True

    def compute_tf(self, doc_index: int) -> Dict[str, float]:
        """
        Calcule TF pour un document

        Args:
            doc_index: Index du document

        Returns:
            Dictionnaire {mot: TF}
        """
        doc = self.documents[doc_index]

        if len(doc) == 0:
            return {}

        # Compter les occurrences
        word_counts = Counter(doc)

        # Normaliser par la longueur du document
        total_words = len(doc)
        tf_dict = {word: count / total_words for word, count in word_counts.items()}

        return tf_dict

    def compute_idf(self) -> Dict[str, float]:
        """
        Calcule IDF pour tout le vocabulaire

        Returns:
            Dictionnaire {mot: IDF}
        """
        n_docs = len(self.documents)
        idf_dict = {}

        for word in self.vocabulary:
            # Compter dans combien de documents le mot apparaît
            doc_count = sum(1 for doc in self.documents if word in doc)

            # Calculer IDF avec log
            # On ajoute 1 au numérateur et dénominateur pour éviter division par zéro
            idf_value = np.log((n_docs + 1) / (doc_count + 1)) + 1

            idf_dict[word] = idf_value

        return idf_dict

    def compute_tfidf(self):
        """
        Combine TF et IDF pour obtenir la matrice TF-IDF
        """
        # Multiplication élément par élément: TF * IDF
        # On broadcaste le vecteur IDF sur toutes les lignes
        self.tfidf_matrix = self.tf_matrix * self.idf_vector

    def vectorize_query(self, query: str) -> np.ndarray:
        """
        Transforme une query en vecteur TF-IDF

        Args:
            query: Texte de la requête

        Returns:
            Vecteur TF-IDF de la query
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté avant de vectoriser une query")

        # Preprocessing de la query
        query_tokens = preprocess_text(query, self.remove_stopwords)

        if len(query_tokens) == 0:
            return np.zeros(len(self.vocabulary))

        # Calculer TF pour la query
        query_counts = Counter(query_tokens)
        total_words = len(query_tokens)

        query_vector = np.zeros(len(self.vocabulary))

        for word, count in query_counts.items():
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                tf_value = count / total_words
                # Appliquer l'IDF du corpus
                query_vector[word_idx] = tf_value * self.idf_vector[word_idx]

        return query_vector

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Recherche les documents les plus similaires à la query

        Args:
            query: Texte de la requête
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de (doc_index, similarity_score) triée par score décroissant
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté avant de faire une recherche")

        # Vectoriser la query
        query_vector = self.vectorize_query(query)

        # Calculer la similarité cosinus avec tous les documents
        similarities = []
        for doc_idx in range(len(self.documents)):
            doc_vector = self.tfidf_matrix[doc_idx]
            similarity = cosine_similarity(query_vector, doc_vector)
            similarities.append((doc_idx, similarity))

        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_explanation(self, query: str, doc_index: int) -> Dict:
        """
        Retourne tous les calculs intermédiaires pour expliquer un score

        Args:
            query: Texte de la requête
            doc_index: Index du document à expliquer

        Returns:
            Dictionnaire avec tous les détails des calculs
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté avant d'obtenir une explication")

        # TF du document
        tf_doc = self.compute_tf(doc_index)

        # TF de la query
        query_tokens = preprocess_text(query, self.remove_stopwords)
        if len(query_tokens) > 0:
            query_counts = Counter(query_tokens)
            total_words = len(query_tokens)
            tf_query = {word: count / total_words for word, count in query_counts.items()}
        else:
            tf_query = {}

        # IDF (seulement pour les mots de la query)
        idf_dict = self.compute_idf()
        relevant_idf = {word: idf_dict[word] for word in tf_query.keys() if word in idf_dict}

        # TF-IDF du document
        tfidf_doc = {}
        for word in self.vocabulary:
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                tfidf_doc[word] = self.tfidf_matrix[doc_index, word_idx]

        # TF-IDF de la query
        query_vector = self.vectorize_query(query)
        tfidf_query = {}
        for word, word_idx in self.word_to_idx.items():
            if query_vector[word_idx] > 0:
                tfidf_query[word] = query_vector[word_idx]

        # Calculs de similarité
        doc_vector = self.tfidf_matrix[doc_index]

        dot_product = np.dot(query_vector, doc_vector)
        norm_doc = np.linalg.norm(doc_vector)
        norm_query = np.linalg.norm(query_vector)

        if norm_doc > 0 and norm_query > 0:
            cosine_sim = dot_product / (norm_doc * norm_query)
        else:
            cosine_sim = 0.0

        return {
            'tf_doc': tf_doc,
            'tf_query': tf_query,
            'idf': relevant_idf,
            'tfidf_doc': tfidf_doc,
            'tfidf_query': tfidf_query,
            'dot_product': float(dot_product),
            'norm_doc': float(norm_doc),
            'norm_query': float(norm_query),
            'cosine_similarity': float(cosine_sim)
        }

    def get_top_words_by_tfidf(self, doc_index: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les mots les plus importants d'un document selon TF-IDF

        Args:
            doc_index: Index du document
            top_k: Nombre de mots à retourner

        Returns:
            Liste de (mot, score_tfidf) triée par score décroissant
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté")

        doc_tfidf = self.tfidf_matrix[doc_index]

        word_scores = [(self.vocabulary[idx], doc_tfidf[idx])
                       for idx in range(len(self.vocabulary))]

        word_scores.sort(key=lambda x: x[1], reverse=True)

        return word_scores[:top_k]

