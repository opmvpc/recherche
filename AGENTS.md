# MISSION: Créer une Application Streamlit Éducative sur TF-IDF

## 🎯 CONTEXTE

Tu es un agent de programmation chargé de créer une application Streamlit pédagogique
pour enseigner les techniques de recherche textuelle à des étudiants en programmation web.

**Phase actuelle:** TF-IDF uniquement (d'autres techniques seront ajoutées plus tard: BM25, embeddings, etc.)

**Public cible:** Étudiants francophones en développement web (niveau bac+2/3)

**Ton objectif:** Créer une application interactive qui explique TF-IDF de manière
complète, claire et imagée, avec des visualisations et des exemples concrets.

---

## 🏗️ ARCHITECTURE DE L'APPLICATION

### Structure de fichiers attendue:

```
tfidf-app/
├── app.py                    # Application Streamlit principale
├── requirements.txt          # Dépendances Python
├── src/
│   ├── __init__.py
│   ├── tfidf_engine.py      # Implémentation TF-IDF from scratch
│   ├── visualizations.py    # Toutes les fonctions de visualisation
│   └── datasets.py          # Chargement et gestion des datasets
├── data/
│   └── .gitkeep            # Les datasets seront téléchargés au runtime
└── README.md               # Documentation du projet
```

### Technologies imposées:

- Python 3.9+
- Streamlit (interface)
- NumPy, Pandas (calculs)
- Matplotlib, Seaborn, Plotly (visualisations)
- scikit-learn (pour comparaison uniquement, pas pour l'implémentation principale)
- requests (pour fetch des datasets)

---

## 📚 CONTENU PÉDAGOGIQUE REQUIS

L'application doit expliquer TF-IDF en suivant cette progression:

### 1. Introduction - Le Problème

- Pourquoi la recherche simple par mots-clés ne suffit pas
- Exemple concret d'un échec de recherche naïve
- Visualisation du problème

### 2. Term Frequency (TF)

- **Intuition:** "Si un mot apparaît souvent, le doc parle de ce sujet"
- **Formule:** `TF(mot, doc) = nombre_occurrences / total_mots_doc`
- **Pourquoi normaliser?** Comparaison doc court vs doc long
- **Exemple calculé étape par étape** (avec 3-4 docs simples)
- **Problème:** Les mots communs ("le", "la") polluent les résultats
- **Visualisation:** Graphique à barres des TF par document

### 3. Inverse Document Frequency (IDF)

- **Intuition:** "Un mot rare est plus informatif qu'un mot commun"
- **Formule:** `IDF(mot) = log(nb_total_docs / nb_docs_contenant_mot)`
- **Pourquoi le log?** Compression de l'échelle (visualiser l'effet)
- **Exemple calculé** avec mots communs vs rares
- **Visualisation:**
  - Courbe IDF en fonction de la fréquence documentaire
  - Word cloud où la taille = IDF

### 4. TF-IDF Combiné

- **Formule:** `TF-IDF = TF × IDF`
- **Signification:** Mots fréquents localement mais rares globalement
- **Calcul complet sur un exemple** (style tableau Excel)
- **Visualisation:** Heatmap TF-IDF (docs en lignes, mots en colonnes)

### 5. Cosine Similarity

- **Intuition géométrique:** Documents = vecteurs, on mesure l'angle
- **Formule:** `cos(θ) = (A·B) / (||A|| × ||B||)`
- **Pourquoi pas juste additionner?** Normalisation par longueur
- **Calcul étape par étape:** dot product, normes, division
- **Visualisation:**
  - Représentation 3D des vecteurs (avec PCA si nécessaire)
  - Matrice de similarité (heatmap)

### 6. Recherche Complète

- Interface de recherche interactive
- Affichage des scores étape par étape
- Comparaison des résultats avec/sans IDF

---

## 🎨 VISUALISATIONS REQUISES

Pour chaque concept, implémente ces visualisations:

### Visualisations TF:

1. **Bar chart:** Fréquence des mots par document
2. **Comparaison:** Doc court vs doc long (avant/après normalisation)

### Visualisations IDF:

1. **Courbe:** IDF en fonction du nombre de documents contenant le mot
2. **Comparaison:** Avec/sans logarithme (montrer l'effet)
3. **Word cloud:** Taille proportionnelle à l'IDF

### Visualisations TF-IDF:

1. **Heatmap:** Matrice complète (docs × mots)
2. **Top mots:** Bar chart des mots les plus importants par document

### Visualisations Similarité:

1. **Scatter plot 3D:** Documents projetés en 3D (interactif avec Plotly)
2. **Heatmap de similarité:** Tous les docs vs tous les docs
3. **Résultats de recherche:** Bar chart des scores de similarité avec la query

### Style des visualisations:

- Palette de couleurs cohérente (ex: "viridis" ou "YlOrRd")
- Annotations sur les graphiques (valeurs, labels)
- Titres explicites et descriptions
- Légendes claires
- Responsive (adaptés à la largeur de l'écran)

---

## 📊 DATASETS EN FRANÇAIS

Utilise des datasets **amusants et variés** en français. Suggestions:

### Dataset 1: Recettes de Cuisine (Léger, ~30 recettes)

```python
# À récupérer via l'API Marmiton ou créer un petit corpus
# Catégories: Italiennes, Asiatiques, Françaises, Mexicaines, etc.
# Permet de tester des queries comme "plat italien", "cuisine épicée"
```

### Dataset 2: Synopsis de Films (Moyen, ~100 films)

```python
# AlloCiné API ou IMDb (traduit)
# Genres variés: Action, Comédie, Horreur, Science-fiction
# Queries: "film drôle", "espace vaisseau", "super-héros"
```

### Dataset 3: Articles Wikipédia FR (Plus gros, ~200 articles)

```python
# Sujets variés via l'API Wikipedia
# Thèmes: Sciences, Histoire, Sport, Technologie, Culture
# Queries: "guerre mondiale", "intelligence artificielle", "football"
```

**Implémentation requise:**

- Fonction de téléchargement avec cache (ne pas retélécharger à chaque run)
- Préprocessing: lowercase, suppression ponctuation basique
- Métadonnées: titre, catégorie, source
- Option de filtrer par catégorie

**Code attendu:**

```python
# src/datasets.py
def load_dataset(name='recettes', use_cache=True, sample_size=None):
    """
    Charge un dataset avec cache

    Args:
        name: 'recettes', 'films', ou 'wikipedia'
        use_cache: Utiliser le cache si disponible
        sample_size: Nombre de docs à charger (None = tous)

    Returns:
        List[Dict]: [{'title': str, 'text': str, 'category': str}, ...]
    """
    pass
```

---

## 🎯 INTERFACE STREAMLIT

### Structure de la page (sidebar + main):

**Sidebar:**

- Sélection du dataset
- Paramètres avancés (optionnels):
  - Taille du dataset
  - Afficher les calculs intermédiaires
  - Thème de couleur des graphiques

**Main Area - Tabs:**

#### Tab 1: 📖 Introduction

- Explication du problème
- Exemple d'échec de recherche naïve
- Présentation de TF-IDF comme solution

#### Tab 2: 🔢 Concepts TF-IDF

**Sous-sections avec st.expander:**

- Term Frequency (TF)
  - Explication théorique
  - Formule avec LaTeX: `st.latex(r"TF = \frac{count}{total}")`
  - Exemple calculé
  - Visualisation
- Inverse Document Frequency (IDF)
  - Même structure
- TF-IDF Combiné
  - Même structure
- Cosine Similarity
  - Même structure

#### Tab 3: 🔍 Recherche Interactive

- Input de recherche (query)
- Bouton "Rechercher"
- Affichage des résultats:
  - Top 5 documents avec scores
  - Snippet du texte (premiers 200 caractères)
  - Option d'afficher le calcul détaillé
- Visualisations:
  - Bar chart des scores
  - Heatmap de similarité (query vs tous les docs)

#### Tab 4: 📊 Exploration du Corpus

- Statistiques du dataset:
  - Nombre de documents
  - Vocabulaire (nombre de mots uniques)
  - Distribution de longueur des documents
- Visualisations globales:
  - Top 20 mots par IDF
  - Matrice TF-IDF complète (heatmap)
  - Projection 2D/3D des documents

#### Tab 5: 🎓 Exemple Pas-à-Pas

- Prendre 3 documents du dataset
- Query prédéfinie
- Dérouler TOUT le calcul étape par étape:
  1. Calcul des TF (tableau)
  2. Calcul des IDF (tableau)
  3. Multiplication → TF-IDF (tableau)
  4. Vectorisation de la query
  5. Cosine similarity (calculs détaillés)
  6. Classement final
- Utiliser des dataframes Pandas pour l'affichage

---

## 💻 IMPLÉMENTATION TF-IDF

**Impératif:** Implémenter TF-IDF **from scratch** (sans utiliser TfidfVectorizer de sklearn)

```python
# src/tfidf_engine.py

class TFIDFEngine:
    """
    Implémentation pédagogique de TF-IDF
    Doit conserver tous les états intermédiaires pour visualisation
    """

    def __init__(self, documents: List[str]):
        """
        Args:
            documents: Liste de textes
        """
        self.documents = documents
        self.vocabulary = None
        self.tf_matrix = None  # Shape: (n_docs, n_vocab)
        self.idf_vector = None  # Shape: (n_vocab,)
        self.tfidf_matrix = None  # Shape: (n_docs, n_vocab)

    def fit(self):
        """Calcule TF, IDF, et TF-IDF pour tous les documents"""
        pass

    def compute_tf(self, doc_index: int) -> Dict[str, float]:
        """Calcule TF pour un document"""
        pass

    def compute_idf(self) -> Dict[str, float]:
        """Calcule IDF pour tout le vocabulaire"""
        pass

    def compute_tfidf(self):
        """Combine TF et IDF"""
        pass

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Recherche les documents les plus similaires

        Returns:
            List of (doc_index, similarity_score) sorted by score desc
        """
        pass

    def get_explanation(self, query: str, doc_index: int) -> Dict:
        """
        Retourne tous les calculs intermédiaires pour expliquer un score

        Returns:
            {
                'tf_doc': Dict[str, float],
                'tf_query': Dict[str, float],
                'idf': Dict[str, float],
                'tfidf_doc': Dict[str, float],
                'tfidf_query': Dict[str, float],
                'dot_product': float,
                'norm_doc': float,
                'norm_query': float,
                'cosine_similarity': float
            }
        """
        pass
```

**Fonctions utilitaires requises:**

```python
def preprocess_text(text: str) -> List[str]:
    """
    Preprocessing simple:
    - Lowercase
    - Suppression ponctuation
    - Split sur espaces
    - Optionnel: suppression stopwords FR
    """
    pass

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcule la similarité cosinus entre deux vecteurs"""
    pass
```

---

## 🎨 GUIDELINES VISUELLES

### Palette de couleurs:

```python
PRIMARY_COLOR = "#1f77b4"      # Bleu
SECONDARY_COLOR = "#ff7f0e"    # Orange
SUCCESS_COLOR = "#2ca02c"      # Vert
WARNING_COLOR = "#d62728"      # Rouge
NEUTRAL_COLOR = "#7f7f7f"      # Gris

# Pour les heatmaps
HEATMAP_COLORSCALE = "YlOrRd"  # Jaune → Orange → Rouge
```

### Style Streamlit:

```python
st.set_page_config(
    page_title="TF-IDF Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Markdown & LaTeX:

- Utiliser `st.latex()` pour les formules mathématiques
- Utiliser des emojis dans les titres (📊, 🔍, 💡, etc.)
- Code blocks avec syntax highlighting
- Callouts avec `st.info()`, `st.success()`, `st.warning()`

### Exemples de texte pédagogique:

```python
st.markdown("""
### 💡 Pourquoi normaliser avec TF?

Imagine deux documents:

- **Doc A (10 mots):** Le mot "chat" apparaît **2 fois**
- **Doc B (100 mots):** Le mot "chat" apparaît **3 fois**

Sans normalisation, Doc B semble plus pertinent (3 > 2).

**Mais!** Doc A consacre 20% de son contenu au mot "chat" (2/10),
tandis que Doc B seulement 3% (3/100).

**Doc A est donc plus "à propos" du chat!** 🎯
""")
```

---

## 🧪 FONCTIONNALITÉS INTERACTIVES

### 1. Comparaison TF vs TF-IDF:

- Toggle pour activer/désactiver l'IDF
- Montrer visuellement la différence de ranking

### 2. Sliders pour paramètres:

```python
# Exemple
min_df = st.slider(
    "Fréquence minimale du document (min_df)",
    min_value=1,
    max_value=10,
    value=1,
    help="Ignorer les mots apparaissant dans moins de X documents"
)
```

### 3. Exemple de query prédéfinies:

```python
example_queries = {
    "recettes": ["plat italien", "cuisine épicée", "dessert chocolat"],
    "films": ["science-fiction espace", "comédie romantique", "super-héros action"],
    "wikipedia": ["guerre mondiale", "intelligence artificielle", "football champion"]
}

selected_example = st.selectbox("Ou choisissez un exemple:", [""] + example_queries[dataset_name])
if selected_example:
    query = selected_example
```

### 4. Export des résultats:

- Bouton pour télécharger les résultats en CSV
- Bouton pour télécharger les visualisations en PNG

---

## 📋 REQUIREMENTS.TXT

Crée un fichier `requirements.txt` avec ces dépendances (versions minimales):

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
scikit-learn>=1.3.0
requests>=2.31.0
scipy>=1.11.0
```

---

## 📝 README.MD

Crée un README complet avec:

1. **Titre et description**
2. **Installation:**

```bash
   pip install -r requirements.txt
```

3. **Lancement:**

```bash
   streamlit run app.py
```

4. **Structure du projet**
5. **Datasets utilisés**
6. **Concepts expliqués**
7. **Captures d'écran** (placeholder pour l'instant)
8. **Auteur et licence**

---

## ⚠️ CONTRAINTES IMPORTANTES

### Tu NE PEUX PAS exécuter de commandes:

- Ne pas utiliser `subprocess`, `os.system()`, ou équivalent
- À la fin de chaque message, liste les commandes que je dois exécuter
- Format:

```bash
  # Commandes à exécuter:
  pip install -r requirements.txt
  streamlit run app.py
```

### Code quality:

- Type hints partout
- Docstrings pour toutes les fonctions
- Comments explicatifs pour la logique complexe
- Gestion des erreurs (try/except)
- Loading spinners (`st.spinner()`) pour les opérations longues

### Performance:

- Utiliser `@st.cache_data` pour le chargement des datasets
- Utiliser `@st.cache_resource` pour l'engine TF-IDF
- Éviter les recalculs inutiles

### UX:

- Messages de chargement clairs
- Gestion des cas d'erreur (dataset vide, query vide, etc.)
- Tooltips (`help=` parameter) pour les éléments complexes
- Progress bars pour les opérations longues

---

## 🎯 LIVRABLES ATTENDUS

À la fin, tu dois avoir créé:

1. ✅ `app.py` - Application Streamlit complète et fonctionnelle
2. ✅ `src/tfidf_engine.py` - Implémentation TF-IDF from scratch
3. ✅ `src/visualizations.py` - Toutes les fonctions de visualisation
4. ✅ `src/datasets.py` - Gestion des datasets
5. ✅ `requirements.txt` - Dépendances
6. ✅ `README.md` - Documentation
7. ✅ Liste des commandes à exécuter

---

## 💬 COMMUNICATION

À chaque message:

1. Explique ce que tu vas faire
2. Crée/modifie les fichiers
3. Termine par une section "📋 Commandes à exécuter" avec toutes les commandes nécessaires

Exemple:

```
📋 Commandes à exécuter:

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py

# (Optionnel) Tester l'import des modules
python -c "from src.tfidf_engine import TFIDFEngine; print('OK')"
```

---

## 🚀 C'EST PARTI!

Commence par créer la structure de base du projet avec `app.py` et `requirements.txt`.
Implémente d'abord le Tab "Introduction" et la structure générale.

Fais-moi un premier jet fonctionnel, même si les datasets ne sont pas encore téléchargés.
On itérera ensuite sur les détails.

**Objectif:** Une application Streamlit qui tourne et affiche au moins la structure des tabs.
