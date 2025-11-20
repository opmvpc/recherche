# Plan de génération des datasets synthétiques

## Objectifs pédagogiques
- Fournir **200 recettes** et **200 synopsis de films** en français avec de vrais paragraphes descriptifs.
- Maximiser la diversité (cuisines, styles culinaires, genres cinématographiques, époques, tons narratifs).
- Stocker les données sous forme de fichiers JSON pour alléger `src/data_loader.py` et faciliter la mise à jour.

## Format technique
- Dossier dédié : `data/synthetic/`
- Fichiers attendus :
  - `recipes_fr.json`
  - `films_fr.json`
- Structure commune à chaque entrée :
  ```json
  {
    "title": "...",
    "text": "...",
    "category": "...",
    "source": "synthetic"
  }
  ```
- Les textes devront inclure au minimum : contexte, éléments clés (ingrédients, personnages, enjeux), variations/catégories.

## Recettes – Axes de diversité
- **Cuisines couvertes** : italienne, française régionale, asiatique (thaï, vietnamienne, japonaise), moyen-orientale, mexicaine, africaine, caraïbéenne, street food, végétarienne/vegan, pâtisserie, fusion moderne, cuisine de fête.
- **Types de plats** : entrées, soupes, salades gourmandes, plats mijotés, grillades, pasta, riz, desserts, boissons chaudes, boulangerie salée.
- **Angles narratifs** : astuces de chef, traditions familiales, version express, version meal-prep, ingrédients de saison, accords mets-vins.
- **Checklist** : limiter à 1 occurrence du même couple (cuisine, plat) pour éviter les répétitions.

## Films – Axes de diversité
- **Genres** : science-fiction, comédie romantique, thriller, polar, animation, drame social, aventure historique, fantastique, horreur psychologique, feel-good, coming-of-age, docu-fiction, super-héros atypiques.
- **Contextes** : différentes périodes (années 20 → futur lointain), multiples lieux (villes européennes, déserts, îles, stations spatiales), diversité des protagonistes.
- **Angles narratifs** : arcs émotionnels, conflits politiques, mystères, road-movies, heists, récits multivers, biopics, films musicaux.
- **Checklist** : varier les structures (voix off, faux documentaire, enquête en puzzle), assurer des synopsis de 3+ phrases.

### Répartition proposée (films)
| Catégorie | Notes |
| --- | --- |
| Science-fiction & anticipation | Chroniques spatiales, dystopies écologiques, techno-thrillers humanistes |
| Comédie romantique & feel-good | Décors urbains variés, personnages contemporains, humour tendre |
| Thriller & polar | Enquêtes à tiroirs, regards sociétaux, twists surprenants |
| Drame social & familial | Protagonistes issus de milieux divers, sujets sensibles |
| Fantastique & mythes | Récits oniriques, folklore, magie réaliste |
| Aventure historique & fresques | Grandes périodes, contextes géopolitiques, road-movies d'époque |
| Animation & jeunesse | Histoires lumineuses mais riches pour public large |
| Horreur & psychologique | Ambiances tendues, enjeux émotionnels, métaphores |
| Super-héros & action | Approches originales, ton européen, questionnement moral |
| Musique, sport & docu-fiction | Biopics inventés, compétitions, coulisses culturelles |

## Guide de variation (recettes) – pour réécrire les 200 entrées
But: chaque recette semble venir d’une source différente. Avant de réécrire, tirer au hasard un style + une structure + un niveau de détail.

- Structures possibles (tirer 1) :
  - Paragraphe libre (sans titres).
  - Etapes numérotées concises.
  - Puces « Ingrédients » puis mini-bloc « Comment faire ».
  - Texte télégraphique façon fiche de service (quantités + mots-clés).
  - Mini-story blog (« J’ai essayé… », « astuce de ma nonna »).
  - Format magazine: chapeau + encadré astuces.
  - Style menu de restaurant (description courte, pas d’étapes).
  - Format « meal-prep » (durées, conservation).
- Ton/voix (tirer 1) :
  - Chef étoilé (précis, vocabulaire technique).
  - Blog familial (chaleureux, adressé au lecteur).
  - Étudiant pressé (rapide, économique).
  - Nutrition/sain (focus substitutions, légèreté).
  - Tradition régionale (références terroir/coutumes).
  - Street food / moderne (punchy, emojis optionnels mais rares).
  - Note de dégustation façon critique gastronomique.
  - Télégraphique « service en cuisine » (impératifs courts).
- Variation contexte/auteur (tirer 1 ou 2) :
  - Cuisine de semaine vs repas de fête.
  - Auteur étudiant, parent pressé, chef de bistrot, nonna, globe-trotter, critique food.
  - Contrainte budget/temps/ustensiles limités ou au contraire équipement pro.
- Détails à varier (tirer 3+ au hasard) :
  - Liste d’ingrédients complète ou partielle.
  - Ustensiles mentionnés ou non.
  - Durées/temps de repos parfois indiqués.
  - Ajout 1-2 astuces (ex: récupération d’eau féculée, substitution).
  - Mention d’accompagnement ou vin, pas systématique.
  - Quantités parfois approximatives (« une bonne poignée ») parfois précises.
  - Avec ou sans unités métriques.
  - Inclusion d’une variante (option végétarienne, sans gluten…).
- Longueur/forme :
  - Tirer une taille cible (mini 1-2 phrases / courte / moyenne / longue).
  - Mélanger phrases complètes et style télégraphique selon la voix.
- Anti-répétition :
  - Pas de structure identique sur deux recettes d’affilée (alterner puces, paragraphes, listes, etc.).
  - Éviter d’ouvrir systématiquement par un libellé suivi d’un « : »; varier ponctuation et tournures.
- Diversifier la granularité :
  - Parfois inclure juste « comment servir » sans étapes.
  - Parfois un micro récit (souvenir, conseil d’ami, astuce de marché).
  - Parfois un bloc « ingrédients » + rien d’autre (style fiche courte).
- Longueur variable :
  - Très court (1-2 phrases) / moyen (5-7 lignes) / plus développé (10-12 phrases).
- Consignes générales :
  - Toujours rester cohérent avec le titre/recette.
  - Éviter la répétition de la même structure sur deux recettes consécutives.
  - Rester en français, ton adapté à la voix choisie.

## Étapes (à cocher)
1. [x] Définir le plan et le format (ce document).
2. [x] Générer les 200 recettes et enregistrer `recipes_fr.json`.
3. [ ] Générer les 200 films et enregistrer `films_fr.json`.
4. [ ] Adapter `src/data_loader.py` pour utiliser ces fichiers.
5. [ ] Prévoir une routine de validation (longueur minimale, unicité des titres).
