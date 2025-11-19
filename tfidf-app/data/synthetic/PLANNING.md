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

## Étapes (à cocher)
1. [x] Définir le plan et le format (ce document).
2. [x] Générer les 200 recettes et enregistrer `recipes_fr.json`.
3. [ ] Générer les 200 films et enregistrer `films_fr.json`.
4. [ ] Adapter `src/data_loader.py` pour utiliser ces fichiers.
5. [ ] Prévoir une routine de validation (longueur minimale, unicité des titres).
