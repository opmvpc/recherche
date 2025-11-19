"""
Script pour télécharger le modèle Sentence-Transformers
À exécuter AVANT de lancer l'application pour éviter le téléchargement lors de la première utilisation
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("📦 TÉLÉCHARGEMENT DU MODÈLE EMBEDDINGS")
print("=" * 60)
print()

try:
    from sentence_transformers import SentenceTransformer

    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    print(f"🎯 Modèle: {model_name}")
    print(f"📏 Taille: ~100-200 MB")
    print(f"🌍 Support: Multilingue (français inclus)")
    print(f"📊 Dimensions: 384")
    print()
    print("⏳ Téléchargement en cours...")
    print("   (Cela peut prendre 2-5 minutes selon votre connexion)")
    print()

    # Téléchargement du modèle
    model = SentenceTransformer(model_name)

    print()
    print("=" * 60)
    print("✅ TÉLÉCHARGEMENT RÉUSSI!")
    print("=" * 60)
    print()
    print(f"📦 Le modèle est maintenant en cache local")
    print(f"🚀 Vous pouvez lancer l'application avec: streamlit run app.py")
    print()

    # Test rapide
    print("🧪 Test du modèle...")
    test_sentence = ["Ceci est une phrase de test"]
    embedding = model.encode(test_sentence)
    print(f"✅ Test réussi! Embedding généré: shape {embedding.shape}")
    print()

except ImportError as e:
    print()
    print("=" * 60)
    print("❌ ERREUR: sentence-transformers n'est pas installé!")
    print("=" * 60)
    print()
    print("📋 Pour installer:")
    print("   pip install sentence-transformers torch transformers")
    print()
    print("💡 Ou installez toutes les dépendances:")
    print("   pip install -r requirements.txt")
    print()
    sys.exit(1)

except Exception as e:
    print()
    print("=" * 60)
    print(f"❌ ERREUR lors du téléchargement: {e}")
    print("=" * 60)
    print()
    print("💡 Solutions possibles:")
    print("   1. Vérifiez votre connexion internet")
    print("   2. Essayez à nouveau (le téléchargement reprendra où il s'est arrêté)")
    print("   3. Vérifiez l'espace disque disponible (~500 MB requis)")
    print()
    sys.exit(1)

