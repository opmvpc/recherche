"""
Chargement des datasets depuis fichiers JSON
Tous les datasets sont versionnés dans Git - pas de fallback!
"""

from typing import List, Dict, Optional
import json
from pathlib import Path


# Dossiers des datasets
SYNTHETIC_DIR = Path("data/synthetic")  # recettes, films
DATASETS_DIR = Path("data/datasets")  # wikipedia


def load_dataset(
    name: str = "recettes",
    extended: bool = False,
    sample_size: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Charge un dataset depuis les fichiers JSON versionnés dans Git

    Args:
        name: 'recettes', 'films', ou 'wikipedia'
        extended: True pour charger la version étendue, False pour la version normale
        sample_size: Limite le nombre de documents (None = tous)

    Returns:
        Liste de documents: [{'title': str, 'text': str, 'category': str}, ...]

    Raises:
        ValueError: Si le dataset n'existe pas
        FileNotFoundError: Si le fichier JSON n'est pas trouvé
    """
    # Mapping nom -> fichier JSON
    dataset_files = {
        "recettes": SYNTHETIC_DIR / "recettes_fr.json",
        "films": SYNTHETIC_DIR / "films_fr.json",
        "wikipedia": DATASETS_DIR / "wikipedia_fr.json",
    }

    if name not in dataset_files:
        raise ValueError(
            f"Dataset '{name}' inconnu. Disponibles: {list(dataset_files.keys())}"
        )

    filepath = dataset_files[name]

    # Vérifier que le fichier existe
    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier {filepath} introuvable!\n"
            f"Les datasets doivent être dans le repo Git.\n"
            f"Vérifie ton clone ou exécute: git pull"
        )

    # Charger le JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[OK] Chargé: {filepath.name} ({len(data)} documents)")

    except json.JSONDecodeError as e:
        raise ValueError(f"Fichier JSON invalide {filepath.name}: {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur lecture {filepath.name}: {e}")

    # Limiter selon extended et sample_size
    if not extended:
        # Mode normal: limites par défaut
        limits = {
            "recettes": 50,
            "films": 30,
            "wikipedia": 200,
        }
        data = data[: limits.get(name, 50)]

    # Sample size custom
    if sample_size and sample_size < len(data):
        data = data[:sample_size]

    print(f"   Retourne: {len(data)} documents")

    return data


def get_dataset_info(name: str) -> Dict[str, any]:
    """
    Retourne les informations d'un dataset

    Args:
        name: Nom du dataset

    Returns:
        Dict avec: name, description, source, sizes (normal, extended)
    """
    infos = {
        "recettes": {
            "name": "Recettes de Cuisine 🍳",
            "description": "Collection de recettes françaises variées (italiennes, asiatiques, françaises, mexicaines, desserts)",
            "source": "Synthétique (généré avec IA)",
            "file": "data/synthetic/recettes_fr.json",
            "size_normal": 50,
            "size_extended": "~1200",
            "categories": [
                "Italienne",
                "Asiatique",
                "Française",
                "Mexicaine",
                "Dessert",
            ],
        },
        "films": {
            "name": "Films & Critiques 🎬",
            "description": "Synopsis et critiques de films populaires en français (action, comédie, drame, SF, horreur)",
            "source": "Synthétique (généré avec IA)",
            "file": "data/synthetic/films_fr.json",
            "size_normal": 30,
            "size_extended": "~1200",
            "categories": ["Action", "Comédie", "Drame", "Science-Fiction", "Horreur"],
        },
        "wikipedia": {
            "name": "Wikipédia FR 📚",
            "description": "Articles Wikipedia français sur des sujets variés (sciences, histoire, géographie, sport, technologie)",
            "source": "Hugging Face (wikimedia/wikipedia)",
            "file": "data/datasets/wikipedia_fr.json",
            "size_normal": 200,
            "size_extended": 1000,
            "categories": [
                "Science",
                "Histoire",
                "Géographie",
                "Sport",
                "Technologie",
                "Art",
                "Général",
            ],
        },
    }

    if name not in infos:
        raise ValueError(f"Dataset '{name}' inconnu")

    return infos[name]


def get_all_datasets_info() -> List[Dict[str, any]]:
    """
    Retourne les infos de tous les datasets disponibles
    """
    return [get_dataset_info(name) for name in ["recettes", "films", "wikipedia"]]


# ============================================================================
# UTILITAIRES (si besoin)
# ============================================================================


def verify_datasets() -> Dict[str, bool]:
    """
    Vérifie que tous les fichiers JSON existent

    Returns:
        Dict {nom_dataset: fichier_existe}
    """
    datasets = {
        "recettes": SYNTHETIC_DIR / "recettes_fr.json",
        "films": SYNTHETIC_DIR / "films_fr.json",
        "wikipedia": DATASETS_DIR / "wikipedia_fr.json",
    }

    results = {}
    for name, filepath in datasets.items():
        exists = filepath.exists()
        results[name] = exists
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {name}: {filepath}")

    return results


if __name__ == "__main__":
    print("=== Vérification des datasets ===\n")
    verify_datasets()

    print("\n=== Test de chargement ===\n")
    for name in ["recettes", "films", "wikipedia"]:
        try:
            data = load_dataset(name, extended=False)
            print(f"[OK] {name}: {len(data)} docs chargés\n")
        except Exception as e:
            print(f"[ERROR] {name}: {e}\n")
