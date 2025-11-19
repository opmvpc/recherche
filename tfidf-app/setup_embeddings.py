"""
Script interactif pour installer et télécharger les dépendances des embeddings
Compatible avec l'application Streamlit
"""

import subprocess
import sys
from pathlib import Path

def check_package_installed(package_name: str) -> bool:
    """Vérifie si un package Python est installé"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_packages():
    """Installe les packages nécessaires pour les embeddings"""
    packages = [
        "sentence-transformers",
        "torch",
        "transformers"
    ]

    print("=" * 60)
    print("📦 INSTALLATION DES DÉPENDANCES EMBEDDINGS")
    print("=" * 60)
    print()

    missing_packages = []
    for package in packages:
        if check_package_installed(package):
            print(f"✅ {package} - déjà installé")
        else:
            print(f"❌ {package} - manquant")
            missing_packages.append(package)

    if not missing_packages:
        print()
        print("✅ Toutes les dépendances sont déjà installées!")
        return True

    print()
    print(f"📋 Packages à installer: {', '.join(missing_packages)}")
    print(f"⏳ Taille estimée: ~500 MB - 1 GB")
    print(f"⌛ Temps estimé: 3-10 minutes")
    print()

    response = input("Voulez-vous installer maintenant? (o/n): ").lower()

    if response != 'o':
        print("❌ Installation annulée.")
        return False

    print()
    print("⏳ Installation en cours...")
    print()

    try:
        # Installation de torch en premier (version CPU par défaut)
        if 'torch' in missing_packages:
            print("📦 Installation de PyTorch (CPU)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
                "--quiet"
            ])
            print("✅ PyTorch installé!")
            missing_packages.remove('torch')

        # Installation des autres packages
        if missing_packages:
            print(f"📦 Installation de {', '.join(missing_packages)}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                *missing_packages,
                "--quiet"
            ])
            print(f"✅ {', '.join(missing_packages)} installés!")

        print()
        print("=" * 60)
        print("✅ INSTALLATION RÉUSSIE!")
        print("=" * 60)
        return True

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"❌ ERREUR lors de l'installation: {e}")
        print("=" * 60)
        print()
        print("💡 Essayez manuellement:")
        print("   pip install sentence-transformers torch transformers")
        return False

def download_model():
    """Télécharge le modèle Sentence-Transformers"""
    try:
        from sentence_transformers import SentenceTransformer

        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

        print()
        print("=" * 60)
        print("📥 TÉLÉCHARGEMENT DU MODÈLE")
        print("=" * 60)
        print()
        print(f"🎯 Modèle: {model_name}")
        print(f"📏 Taille: ~100-200 MB")
        print(f"🌍 Support: Multilingue (français inclus)")
        print()
        print("⏳ Téléchargement en cours...")
        print("   (Première utilisation: 2-5 minutes)")
        print()

        # Téléchargement
        model = SentenceTransformer(model_name)

        print()
        print("=" * 60)
        print("✅ MODÈLE TÉLÉCHARGÉ!")
        print("=" * 60)
        print()

        # Test
        print("🧪 Test du modèle...")
        test = ["Test en français"]
        embedding = model.encode(test)
        print(f"✅ Test réussi! Shape: {embedding.shape}")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERREUR: {e}")
        print("=" * 60)
        return False

def main():
    """Fonction principale"""
    print()
    print("🔍 SETUP EMBEDDINGS POUR L'APPLICATION STREAMLIT")
    print()

    # 1. Installation des packages
    if not install_packages():
        sys.exit(1)

    # 2. Téléchargement du modèle
    if not download_model():
        sys.exit(1)

    # 3. Succès!
    print()
    print("=" * 60)
    print("🎉 CONFIGURATION TERMINÉE!")
    print("=" * 60)
    print()
    print("🚀 Vous pouvez maintenant lancer l'application:")
    print("   streamlit run app.py")
    print()
    print("📊 Les sections Embeddings et Synthèse seront débloquées!")
    print()

if __name__ == "__main__":
    main()

