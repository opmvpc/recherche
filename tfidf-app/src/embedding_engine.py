"""
Moteur de Recherche par Embeddings Vectoriels
Utilise Sentence-Transformers pour la recherche sémantique
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import os


def download_model_with_progress(model_name: str, cache_folder: str = None):
    """
    Télécharge un modèle Sentence-Transformers avec feedback

    Args:
        model_name: Nom du modèle HuggingFace
        cache_folder: Dossier de cache (optionnel)

    Returns:
        Le modèle chargé
    """
    try:
        # Essayer de charger (télécharge si pas en cache)
        print(f"📥 Téléchargement du modèle {model_name}...")
        print("⏳ Première utilisation: ~100-200 MB à télécharger...")
        print("📦 Le modèle sera mis en cache pour les prochaines utilisations!")

        model = SentenceTransformer(model_name, cache_folder=cache_folder)

        print(f"✅ Modèle {model_name} chargé avec succès!")
        return model

    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        raise


class EmbeddingSearch:
    """
    Implémentation de recherche sémantique avec embeddings
    Utilise des modèles pré-entraînés de Sentence-BERT
    """

    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        cache_dir: Optional[str] = './data/cache',
        model_cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: Nom du modèle HuggingFace Sentence-Transformers
            cache_dir: Dossier pour cache des embeddings calculés
            model_cache_dir: Dossier pour cache du modèle téléchargé
        """
        self.model_name = model_name

        # Téléchargement automatique du modèle avec feedback
        self.model = download_model_with_progress(model_name, cache_folder=model_cache_dir)

        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.documents = []
        self.embeddings = None
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def index(
        self,
        documents: List[str],
        use_cache: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ):
        """
        Index les documents en calculant leurs embeddings

        Args:
            documents: Liste de textes à indexer
            use_cache: Utiliser le cache si disponible
            batch_size: Taille des batches pour encoding
            show_progress: Afficher barre de progression
        """
        self.documents = documents

        # Vérifier cache
        if use_cache and self.cache_dir:
            cached_embeddings = self._load_cache()
            if cached_embeddings is not None:
                self.embeddings = cached_embeddings
                return

        # Calculer embeddings
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        # Sauvegarder cache
        if use_cache and self.cache_dir:
            self._save_cache()

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recherche les documents les plus similaires sémantiquement

        Args:
            query: Requête texte
            top_k: Nombre de résultats à retourner

        Returns:
            List of {'index': int, 'score': float, 'document': str}
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call .index() first.")

        # Encoder query
        query_embedding = self.model.encode([query])[0]

        # Calcul similarités cosinus
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        # Top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx]),
                'document': self.documents[idx]
            })

        return results

    def find_similar(
        self,
        doc_index: int,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Trouve les documents similaires à un document donné

        Args:
            doc_index: Index du document source
            top_k: Nombre de résultats
            exclude_self: Exclure le document lui-même des résultats

        Returns:
            List of {'index': int, 'score': float, 'document': str}
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call .index() first.")

        doc_embedding = self.embeddings[doc_index]

        similarities = cosine_similarity(
            [doc_embedding],
            self.embeddings
        )[0]

        if exclude_self:
            similarities[doc_index] = -1

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] < 0:  # Skip excluded
                continue
            results.append({
                'index': int(idx),
                'score': float(similarities[idx]),
                'document': self.documents[idx]
            })

        return results

    def get_similarity_matrix(self) -> np.ndarray:
        """
        Calcule la matrice de similarité complète entre tous les documents

        Returns:
            Matrice numpy (n_docs × n_docs)
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call .index() first.")

        return cosine_similarity(self.embeddings, self.embeddings)

    def get_embeddings(self) -> np.ndarray:
        """Retourne les embeddings calculés"""
        return self.embeddings

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Encode une query en vecteur"""
        return self.model.encode([query])[0]

    def _load_cache(self) -> Optional[np.ndarray]:
        """Charge les embeddings depuis le cache"""
        import hashlib

        # Hash des documents pour identifier le cache
        doc_hash = hashlib.md5(
            "".join(self.documents).encode()
        ).hexdigest()[:8]

        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{doc_hash}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def _save_cache(self):
        """Sauvegarde les embeddings dans le cache"""
        import hashlib

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            doc_hash = hashlib.md5(
                "".join(self.documents).encode()
            ).hexdigest()[:8]

            cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{doc_hash}.pkl"

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.embeddings, f)
            except Exception:
                pass  # Cache failed, not critical
