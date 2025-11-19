"""
Script pour télécharger le dataset Wikipedia FR (1000 articles).
Le fichier généré sera versionné dans git pour un téléchargement rapide via GitHub.

Usage:
    python download_datasets.py
"""

import json
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
    print("✅ Hugging Face datasets est disponible!")
except ImportError:
    HF_AVAILABLE = False
    print("❌ Hugging Face datasets n'est PAS installé!")
    print("   Installe-le avec: pip install datasets")
    exit(1)


# Créer les dossiers nécessaires
DATA_DIR = Path("data/datasets")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Dossier de données: {DATA_DIR.absolute()}\n")


def save_dataset(data: List[Dict], filepath: Path):
    """Sauvegarde un dataset au format JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Sauvegardé: {filepath.name} ({len(data)} documents)")


def download_wikipedia(target_size: int = 1000):
    """
    Télécharge des articles Wikipedia FR variés

    Args:
        target_size: Nombre d'articles à télécharger
    """
    print(f"📚 Téléchargement Wikipedia FR ({target_size} articles)...")

    try:
        # Charger Wikipedia FR avec streaming
        dataset = hf_load_dataset(
            "wikimedia/wikipedia",
            "20231101.fr",  # Dump français de novembre 2023
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        # Shuffle pour avoir de la diversité
        dataset = dataset.shuffle(seed=42, buffer_size=10000)

        articles = []
        seen_titles = set()

        for article in dataset:
            title = article.get('title', '')
            text = article.get('text', '')

            # Skip si trop court ou doublon
            if len(text) < 200 or title in seen_titles:
                continue

            # Limiter la longueur
            if len(text) > 2000:
                text = text[:2000] + "..."

            # Deviner la catégorie (basique)
            category = "Général"
            if any(word in title.lower() for word in ['guerre', 'bataille', 'conflit']):
                category = "Histoire"
            elif any(word in title.lower() for word in ['science', 'physique', 'chimie', 'biologie']):
                category = "Sciences"
            elif any(word in title.lower() for word in ['sport', 'football', 'rugby', 'tennis']):
                category = "Sport"
            elif any(word in title.lower() for word in ['film', 'cinéma', 'musique', 'art']):
                category = "Culture"

            articles.append({
                'title': title,
                'text': text,
                'category': category
            })

            seen_titles.add(title)

            if len(articles) >= target_size:
                break

            # Progress
            if len(articles) % 100 == 0:
                print(f"   📥 {len(articles)}/{target_size} articles...")

        # Sauvegarder
        save_dataset(articles, DATA_DIR / "wikipedia_fr.json")
        return True

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


# Dataset LIVRES supprimé (trop lourd: 400 MB)


def main():
    """Télécharge le dataset Wikipedia FR"""
    print("=" * 60)
    print("🔥 TÉLÉCHARGEMENT WIKIPEDIA FR (1000 articles)")
    print("=" * 60)
    print()

    # Vérifier HuggingFace
    if not HF_AVAILABLE:
        print("❌ Installation requise: pip install datasets")
        return

    # === WIKIPEDIA (1000 articles) ===
    success = download_wikipedia(target_size=1000)
    print()

    # === RÉSUMÉ ===
    print("=" * 60)
    if success:
        print("✅ SUCCÈS: Wikipedia FR téléchargé!")
        print()
        print("📋 Fichier créé:")
        wiki_file = DATA_DIR / "wikipedia_fr.json"
        if wiki_file.exists():
            size_mb = wiki_file.stat().st_size / (1024 * 1024)
            print(f"   - {wiki_file.name} ({size_mb:.2f} MB)")
        print()
        print("📝 Ce fichier sera versionné dans git pour un téléchargement rapide!")
        print()
        print("🚀 Tu peux maintenant lancer l'app Streamlit!")
        print("   streamlit run app.py")
    else:
        print("❌ ÉCHEC: Vérifie les erreurs ci-dessus et réessaie.")
    print("=" * 60)


if __name__ == "__main__":
    main()
