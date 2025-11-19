# 📦 Gestion des Datasets

## 🎯 Système Simplifié (Ultra Rapide!)

Les datasets sont maintenant **versionnés dans git** pour un téléchargement instantané via GitHub!

**Wikipedia FR (1000 articles)** → Inclus dans le repo git (~3 MB)
**Recettes/Films synthétiques** → Inclus dans le repo git

---

## 🚀 Installation Rapide

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. C'est tout!

Le fichier `wikipedia_fr.json` est déjà dans le repo!
**Pas besoin de télécharger**, **pas besoin de token HF**!

---

## 🔄 (Optionnel) Regénérer Wikipedia

Si tu veux regénérer le dataset Wikipedia (par exemple pour avoir des articles plus récents):

```bash
python download_datasets.py
```

**Durée:** ~2-3 minutes
**Espace disque:** ~3 MB

---

## 📂 Structure des Fichiers

```
tfidf-app/
├── data/
│   ├── datasets/               # Datasets (versionnés dans git)
│   │   └── wikipedia_fr.json   # 1000 articles Wikipedia (~3 MB)
│   ├── synthetic/              # Datasets synthétiques (versionnés)
│   │   ├── recipes_fr.json     # ~1200 recettes
│   │   └── films_fr.json       # ~1200 films
│   └── cache/                  # Cache TF-IDF/BM25 (ignoré par git)
└── download_datasets.py        # Script pour regénérer Wikipedia
```

---

## 📊 Datasets Disponibles

### Mode Normal (petits datasets)

- **Recettes:** 50 docs depuis `data/synthetic/recipes_fr.json`
- **Films:** 50 docs depuis `data/synthetic/films_fr.json`
- **Wikipedia:** 27 docs hardcodés (fallback)

### Mode Étendu (gros datasets)

- **Recettes:** ~1200 docs depuis `data/synthetic/recipes_fr.json`
- **Films:** ~1200 docs depuis `data/synthetic/films_fr.json`
- **Wikipedia:** 1000 docs depuis `data/datasets/wikipedia_fr.json` (versionné dans git!)

---

## ⚠️ Dépannage

### Erreur: "Fichier non trouvé"

```
⚠️ Fichier wikipedia_fr.json non trouvé!
   Normalement versionné dans git, vérifie ton clone!
```

**Solution:**
1. Vérifie que tu as bien cloné le repo
2. Ou regénère le fichier:
```bash
python download_datasets.py
```

---

### Les datasets ne se chargent pas

**1. Vérifie que le fichier JSON existe:**
```bash
ls data/datasets/
```

Tu devrais voir:
- `wikipedia_fr.json`

**2. Si absent, regénère-le:**
```bash
python download_datasets.py
```

---

## 🎯 Workflow Complet

```bash
# 1. Cloner le repo (Wikipedia déjà inclus!)
git clone <url_du_repo>
cd tfidf-app

# 2. Installation
pip install -r requirements.txt

# 3. Lancer l'app
streamlit run app.py
```

---

## 📈 Avantages du Système Actuel

✅ **Aucun téléchargement:** Wikipedia déjà dans le repo!
✅ **Rapide:** Clone via GitHub ultra rapide
✅ **Offline:** Fonctionne sans connexion
✅ **Contrôle:** Datasets versionnés et reproductibles
✅ **Simple:** Pas de configuration, pas de token HF

---

## 🔄 Mise à Jour des Datasets

Pour mettre à jour Wikipedia avec des articles plus récents:

```bash
# Regénérer Wikipedia
python download_datasets.py

# Commit la nouvelle version
git add data/datasets/wikipedia_fr.json
git commit -m "Update Wikipedia dataset"
```

---

## 📝 Notes

- `wikipedia_fr.json` est **versionné dans git** (exception dans `.gitignore`)
- Le cache TF-IDF/BM25 (`data/cache/`) est ignoré par git
- Le cache des embeddings (`models/`, `.cache/`) est ignoré par git
- Les datasets synthétiques (`data/synthetic/`) sont versionnés dans git
