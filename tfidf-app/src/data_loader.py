"""
Chargement et gestion des datasets en français
Utilise Hugging Face pour de VRAIS datasets (1k-10k docs)
Avec fallback vers données hardcodées si offline
"""

from typing import List, Dict, Optional
import json
import os
from pathlib import Path
import random
import pickle

# Try to import Hugging Face datasets
try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
    print("✅ Hugging Face 'datasets' importé avec succès!")
except ImportError as e:
    HF_AVAILABLE = False
    print(f"⚠️ Hugging Face 'datasets' non disponible: {e}")
    print("   Utilisation des données hardcodées.")
except Exception as e:
    HF_AVAILABLE = False
    print(f"❌ Erreur lors de l'import de 'datasets': {type(e).__name__}: {e}")
    print("   Utilisation des données hardcodées.")

# Dossier des datasets téléchargés localement (via download_datasets.py)
DATASETS_DIR = Path("data/datasets")

# Datasets hardcodés pour le MVP (pas besoin d'API externe)
RECETTES_DATA = [
    {
        "title": "Pâtes Carbonara",
        "text": "Recette italienne traditionnelle avec des oeufs, du parmesan, du guanciale et du poivre noir. Faire cuire les pâtes al dente. Mélanger les jaunes d'oeufs avec le parmesan râpé. Faire revenir le guanciale coupé en lardons. Mélanger le tout hors du feu pour obtenir une sauce crémeuse sans oeufs brouillés.",
        "category": "Italienne",
    },
    {
        "title": "Risotto aux Champignons",
        "text": "Plat italien crémeux à base de riz arborio, bouillon de légumes, champignons frais, parmesan, vin blanc et beurre. Faire revenir l'oignon dans le beurre, ajouter le riz et nacrer. Déglacer au vin blanc puis ajouter le bouillon louche par louche en remuant constamment.",
        "category": "Italienne",
    },
    {
        "title": "Pizza Margherita",
        "text": "Pizza italienne classique avec pâte maison, sauce tomate, mozzarella di bufala, basilic frais et huile d'olive. Étaler la pâte finement, napper de sauce tomate, ajouter la mozzarella en morceaux. Cuire au four très chaud. Ajouter le basilic frais à la sortie du four.",
        "category": "Italienne",
    },
    {
        "title": "Tiramisu",
        "text": "Dessert italien avec mascarpone, biscuits imbibés de café, cacao amer et oeufs. Monter les jaunes avec le sucre et le mascarpone. Battre les blancs en neige et incorporer délicatement. Tremper les biscuits dans le café fort et monter en couches alternées. Saupoudrer de cacao et réserver au frais.",
        "category": "Italienne",
    },
    {
        "title": "Pad Thaï",
        "text": "Nouilles de riz sautées thaïlandaises avec crevettes, oeufs, cacahuètes, pousses de soja, sauce tamarind et sauce poisson. Faire tremper les nouilles. Sauter les crevettes, ajouter les nouilles et la sauce. Incorporer les oeufs brouillés. Garnir de cacahuètes concassées et coriandre.",
        "category": "Asiatique",
    },
    {
        "title": "Ramen Japonais",
        "text": "Soupe de nouilles japonaise avec bouillon miso, nouilles ramen, porc chashu, oeuf mollet, algues nori, oignons verts et maïs. Préparer un bouillon riche pendant plusieurs heures. Cuire les nouilles al dente. Assembler avec les garnitures.",
        "category": "Asiatique",
    },
    {
        "title": "Curry Vert Thaï",
        "text": "Curry thaïlandais épicé avec pâte de curry vert, lait de coco, poulet, aubergines, basilic thaï, citronnelle et galanga. Faire revenir la pâte de curry, ajouter le lait de coco et laisser mijoter. Ajouter les légumes et le poulet. Servir avec du riz jasmin.",
        "category": "Asiatique",
    },
    {
        "title": "Sushi Maki",
        "text": "Rouleaux de riz vinaigré japonais avec poisson cru, avocat, concombre, algue nori. Préparer le riz à sushi avec vinaigre de riz, sucre et sel. Étaler sur la nori, garnir de poisson et légumes. Rouler fermement avec un makisu et découper.",
        "category": "Asiatique",
    },
    {
        "title": "Bo Bun",
        "text": "Salade vietnamienne fraîche avec vermicelles de riz, boeuf mariné, salade, herbes aromatiques, cacahuètes et sauce nuoc mam. Faire mariner le boeuf dans une sauce sucrée-salée. Griller le boeuf. Assembler avec les vermicelles froids, salade et herbes.",
        "category": "Asiatique",
    },
    {
        "title": "Boeuf Bourguignon",
        "text": "Plat mijoté français traditionnel avec boeuf braisé dans du vin rouge de Bourgogne, lardons, champignons, oignons grelots et carottes. Faire mariner la viande dans le vin rouge. Saisir la viande, ajouter les légumes et le vin. Mijoter doucement pendant trois heures.",
        "category": "Française",
    },
    {
        "title": "Coq au Vin",
        "text": "Poulet mijoté dans du vin rouge avec lardons, champignons, oignons et bouquet garni. Faire flamber le coq au cognac. Ajouter le vin rouge et laisser mijoter longuement. Servir avec des pommes de terre vapeur ou des pâtes fraîches.",
        "category": "Française",
    },
    {
        "title": "Quiche Lorraine",
        "text": "Tarte salée française avec pâte brisée, lardons fumés, oeufs, crème fraîche et gruyère râpé. Foncer un moule avec la pâte. Faire revenir les lardons. Battre les oeufs avec la crème. Disposer les lardons et verser l'appareil. Cuire au four jusqu'à coloration dorée.",
        "category": "Française",
    },
    {
        "title": "Ratatouille",
        "text": "Plat provençal de légumes mijotés: aubergines, courgettes, poivrons, tomates, oignons, ail, herbes de Provence et huile d'olive. Faire revenir chaque légume séparément. Assembler et laisser mijoter doucement. Servir chaud ou froid avec du pain de campagne.",
        "category": "Française",
    },
    {
        "title": "Crêpes Suzette",
        "text": "Dessert français flambé avec crêpes fines, beurre d'agrumes, jus d'orange, zeste, sucre et Grand Marnier. Préparer des crêpes fines. Préparer le beurre d'orange. Faire chauffer les crêpes dans le beurre sucré et flamber au Grand Marnier devant les convives.",
        "category": "Française",
    },
    {
        "title": "Tacos al Pastor",
        "text": "Tacos mexicains avec porc mariné aux épices, ananas grillé, coriandre, oignon et tortillas de maïs. Faire mariner le porc avec des épices mexicaines et du jus d'ananas. Griller le porc et l'ananas. Servir dans des tortillas chaudes avec coriandre et oignon.",
        "category": "Mexicaine",
    },
    {
        "title": "Guacamole",
        "text": "Sauce mexicaine à base d'avocats écrasés, citron vert, oignon rouge, tomate, coriandre, piment jalapeño et sel. Écraser les avocats à la fourchette en gardant des morceaux. Ajouter tous les ingrédients finement hachés. Mélanger délicatement et servir immédiatement.",
        "category": "Mexicaine",
    },
    {
        "title": "Enchiladas",
        "text": "Tortillas de maïs farcies mexicaines avec poulet effiloché, sauce chili rouge, fromage fondu et crème aigre. Pocher le poulet et l'effilocher. Garnir les tortillas de poulet et rouler. Napper de sauce chili, couvrir de fromage et gratiner au four.",
        "category": "Mexicaine",
    },
    {
        "title": "Chili Con Carne",
        "text": "Ragoût épicé mexicain avec boeuf haché, haricots rouges, tomates, oignons, poivrons, piment chili, cumin et paprika fumé. Faire revenir le boeuf et les oignons. Ajouter les épices, les tomates et les haricots. Laisser mijoter longuement pour développer les saveurs.",
        "category": "Mexicaine",
    },
    {
        "title": "Paella Valenciana",
        "text": "Plat espagnol de riz au safran avec poulet, lapin, haricots verts, poivrons rouges et romarin. Faire revenir les viandes dans une grande poêle. Ajouter les légumes et le riz. Mouiller avec du bouillon au safran. Cuire sans remuer jusqu'à formation du socarrat.",
        "category": "Espagnole",
    },
    {
        "title": "Gaspacho",
        "text": "Soupe froide espagnole à base de tomates crues, concombre, poivron, ail, vinaigre de xérès et huile d'olive. Mixer tous les légumes avec le pain rassis. Assaisonner avec le vinaigre et l'huile. Réserver au réfrigérateur et servir très froid avec des croûtons.",
        "category": "Espagnole",
    },
    {
        "title": "Moussaka",
        "text": "Gratin grec avec aubergines, viande hachée d'agneau, sauce tomate épicée, béchamel et fromage. Faire revenir l'agneau avec les tomates et les épices. Griller les tranches d'aubergines. Monter en couches alternées. Napper de béchamel et gratiner au four.",
        "category": "Grecque",
    },
    {
        "title": "Poulet Tikka Masala",
        "text": "Curry indien avec morceaux de poulet marinés au yaourt et épices, sauce tomate crémeuse au garam masala, gingembre, ail et crème. Faire mariner le poulet dans yaourt et épices. Griller le poulet. Préparer la sauce tomate-crème épicée. Mijoter le poulet dans la sauce.",
        "category": "Indienne",
    },
    {
        "title": "Biryani",
        "text": "Riz indien parfumé aux épices avec agneau ou poulet, oignons caramélisés, safran, cardamome, cannelle et menthe fraîche. Faire mariner la viande. Cuire le riz avec les épices. Monter en couches avec la viande et les oignons frits. Cuire à l'étouffée.",
        "category": "Indienne",
    },
    {
        "title": "Fondue Savoyarde",
        "text": "Plat convivial savoyard de fromages fondus: Comté, Beaufort, Gruyère, vin blanc sec, ail et kirsch. Frotter le caquelon avec l'ail. Faire fondre les fromages râpés avec le vin blanc. Ajouter le kirsch. Servir avec des cubes de pain rassis et des pommes de terre.",
        "category": "Française",
    },
    {
        "title": "Tartiflette",
        "text": "Gratin savoyard avec pommes de terre, reblochon, lardons fumés, oignons et vin blanc. Faire cuire les pommes de terre. Faire revenir les lardons et oignons. Monter en couches dans un plat. Poser le reblochon coupé en deux sur le dessus. Gratiner jusqu'à ce que le fromage soit fondu.",
        "category": "Française",
    },
    {
        "title": "Fish and Chips",
        "text": "Plat britannique de poisson pané frit avec frites épaisses, sauce tartare et purée de petits pois. Préparer une pâte à frire à la bière. Tremper le poisson dans la pâte et frire. Couper des grosses frites et frire deux fois. Servir très chaud avec du vinaigre de malt.",
        "category": "Britannique",
    },
    {
        "title": "Falafel",
        "text": "Boulettes végétariennes du Moyen-Orient à base de pois chiches, persil, coriandre, oignon, ail et cumin. Mixer les pois chiches avec les herbes et épices. Former des boulettes et frire dans l'huile chaude. Servir dans un pain pita avec sauce tahini, salade et pickles.",
        "category": "Moyen-Orient",
    },
    {
        "title": "Houmous",
        "text": "Purée de pois chiches orientale avec tahini, citron, ail, huile d'olive et cumin. Mixer les pois chiches cuits avec le tahini, jus de citron et ail. Ajouter de l'eau pour la texture. Servir avec un filet d'huile d'olive et du paprika. Accompagner de pain pita chaud.",
        "category": "Moyen-Orient",
    },
    {
        "title": "Salade César",
        "text": "Salade romaine avec poulet grillé, croûtons aillés, parmesan, sauce césar crémeuse à base d'anchois, ail, jaune d'oeuf, moutarde et huile d'olive. Préparer la sauce en émulsionnant tous les ingrédients. Griller le poulet. Mélanger la salade avec la sauce, ajouter le poulet et les croûtons.",
        "category": "Américaine",
    },
    {
        "title": "Burger Maison",
        "text": "Hamburger fait maison avec pain brioché, steak haché de boeuf, fromage cheddar fondu, salade iceberg, tomate, oignon, cornichons et sauce barbecue. Assaisonner généreusement les steaks. Griller les pains. Cuire les steaks et faire fondre le fromage. Assembler avec les garnitures.",
        "category": "Américaine",
    },
]

FILMS_DATA = [
    {
        "title": "Interstellar",
        "text": "Film de science-fiction épique où des astronautes traversent un trou de ver spatial pour trouver une nouvelle planète habitable pour l'humanité. Voyage interstellaire, relativité du temps, trou noir, dimensions parallèles, amour transcendant l'espace-temps.",
        "category": "Science-fiction",
    },
    {
        "title": "Inception",
        "text": "Thriller psychologique où des espions pénètrent dans les rêves pour voler des secrets. Architecture de rêves, réalité vs illusion, toupie, niveaux de conscience imbriqués, fin ambiguë.",
        "category": "Science-fiction",
    },
    {
        "title": "Matrix",
        "text": "Film cyberpunk révolutionnaire où l'humanité découvre que leur réalité est une simulation informatique. Néo l'élu, pilule rouge, combat contre les machines, kung-fu, effets bullet time iconiques.",
        "category": "Science-fiction",
    },
    {
        "title": "Blade Runner 2049",
        "text": "Suite atmosphérique sur un blade runner réplicant qui découvre un secret enfoui. Futur dystopique, pluie acide, réplicants, questions d'humanité, photographie époustouflante, Denis Villeneuve.",
        "category": "Science-fiction",
    },
    {
        "title": "Arrival",
        "text": "Film de science-fiction contemplatif sur une linguiste qui apprend le langage d'extraterrestres pour comprendre leur mission sur Terre. Communication non-linéaire, cercles de langage alien, temps non-linéaire.",
        "category": "Science-fiction",
    },
    {
        "title": "Le Seigneur des Anneaux",
        "text": "Trilogie épique fantasy où un hobbit doit détruire un anneau maléfique pour sauver la Terre du Milieu. Quête héroïque, elfes, nains, magiciens, batailles épiques, Gollum, Frodon, Aragorn.",
        "category": "Fantasy",
    },
    {
        "title": "Harry Potter à l'école des sorciers",
        "text": "Premier film de la saga magique où un jeune sorcier découvre son héritage et entre à l'école de magie de Poudlard. Baguettes magiques, Quidditch, Voldemort, amitié, cours de potions.",
        "category": "Fantasy",
    },
    {
        "title": "Le Hobbit",
        "text": "Préquelle du Seigneur des Anneaux où Bilbo Sacquet part à l'aventure avec des nains pour récupérer leur trésor gardé par un dragon. Smaug le dragon, Gollum, énigmes, montagne solitaire.",
        "category": "Fantasy",
    },
    {
        "title": "The Dark Knight",
        "text": "Film de super-héros sombre où Batman affronte le Joker qui sème le chaos à Gotham City. Heath Ledger iconique, dilemmes moraux, Harvey Dent, explosion d'hôpital, camion qui se retourne.",
        "category": "Action",
    },
    {
        "title": "Avengers Endgame",
        "text": "Film culminant de l'univers Marvel où les super-héros voyagent dans le temps pour inverser le snap de Thanos. Voyage temporel, pierres d'infinité, sacrifice de Tony Stark, bataille finale épique.",
        "category": "Action",
    },
    {
        "title": "Mad Max Fury Road",
        "text": "Film d'action post-apocalyptique explosif avec des courses-poursuites de véhicules dans le désert. Impératrice Furiosa, guitariste lance-flammes, cascades pratiques, désert, essence, eau.",
        "category": "Action",
    },
    {
        "title": "John Wick",
        "text": "Film d'action stylisé où un tueur à gages légendaire sort de sa retraite pour venger son chien. Chorégraphie de combat, hotel Continental, costumes élégants, vengeance, sous-monde criminel.",
        "category": "Action",
    },
    {
        "title": "Mission Impossible",
        "text": "Série de films d'espionnage avec cascades incroyables et missions impossibles. Tom Cruise qui court, escalade, saute, masques en silicone, gadgets high-tech, IMF.",
        "category": "Action",
    },
    {
        "title": "La La Land",
        "text": "Comédie musicale romantique moderne sur deux artistes à Los Angeles qui tombent amoureux tout en poursuivant leurs rêves. Jazz, danse, auditions, amour vs ambition, fin douce-amère.",
        "category": "Romance",
    },
    {
        "title": "Titanic",
        "text": "Romance tragique sur le paquebot qui coule, entre un artiste pauvre et une aristocrate. Iceberg, Je suis le roi du monde, scène du dessin, porte qui flotte, diamant du coeur de l'océan.",
        "category": "Romance",
    },
    {
        "title": "Nos Jours Heureux",
        "text": "Comédie française sur un camp de vacances avec des animateurs loufoques et des enfants attachants. Colonie de vacances, été, jeux, chansons, amitié, nostalgie.",
        "category": "Comédie",
    },
    {
        "title": "Les Bronzés",
        "text": "Comédie française culte sur des vacanciers au club Méditerranée en Afrique. Jean-Claude Dusse, Popeye, animations débiles, plage, soleil, répliques cultes.",
        "category": "Comédie",
    },
    {
        "title": "Le Parrain",
        "text": "Film de mafia classique sur la famille Corleone à New York. Don Vito, Michael qui refuse puis devient parrain, offre qu'on ne peut refuser, cheval, baptême avec meurtres.",
        "category": "Drame",
    },
    {
        "title": "Intouchables",
        "text": "Comédie dramatique française sur l'amitié entre un aristocrate tétraplégique et son aide-soignant de banlieue. Handicap, amitié improbable, humour, paragliding, Omar Sy, François Cluzet.",
        "category": "Comédie",
    },
    {
        "title": "La Liste de Schindler",
        "text": "Film historique dramatique en noir et blanc sur Oskar Schindler qui sauve des Juifs pendant l'Holocauste. Seconde guerre mondiale, camps de concentration, liste, manteau rouge, humanité.",
        "category": "Drame",
    },
    {
        "title": "Forrest Gump",
        "text": "Drame feel-good où un homme simple d'esprit traverse les grands événements du 20ème siècle américain. Boîte de chocolats, courir, Jenny, crevettes, banc, plume.",
        "category": "Drame",
    },
    {
        "title": "Psychose",
        "text": "Thriller psychologique d'Hitchcock avec la scène de douche iconique. Motel Bates, Norman Bates, mère dans le rocking chair, couteau, rideau de douche, twist final.",
        "category": "Horreur",
    },
    {
        "title": "Shining",
        "text": "Film d'horreur psychologique de Kubrick dans un hotel isolé en hiver. Here's Johnny, jumelles, labyrinthe de neige, Room 237, machine à écrire, All work and no play.",
        "category": "Horreur",
    },
    {
        "title": "L'Exorciste",
        "text": "Film d'horreur classique sur une possession démoniaque d'une fillette. Tête qui tourne, vomissure verte, prêtre exorciste, escaliers, voix démoniaque.",
        "category": "Horreur",
    },
    {
        "title": "Alien",
        "text": "Film d'horreur spatial où une créature extraterrestre traque l'équipage d'un vaisseau. Xenomorphe, chestburster, Ripley, vaisseau spatial Nostromo, In space no one can hear you scream.",
        "category": "Horreur",
    },
    {
        "title": "Indiana Jones",
        "text": "Série d'aventures avec un archéologue baroudeur qui cherche des reliques mystiques. Fouet, chapeau, serpents, arche perdue, graal, boule géante, Nazis.",
        "category": "Aventure",
    },
    {
        "title": "Jurassic Park",
        "text": "Film d'aventure sur un parc d'attractions avec des dinosaures clonés qui se rebelle. T-Rex, vélociraptor intelligent, ADN dans l'ambre, Jeff Goldblum, Life finds a way.",
        "category": "Aventure",
    },
    {
        "title": "Pirates des Caraïbes",
        "text": "Film d'aventure fantasy sur des pirates avec Jack Sparrow excentrique. Malédiction, perle noire, kraken, Davy Jones, compas magique, rhum, Keith Richards.",
        "category": "Aventure",
    },
    {
        "title": "Toy Story",
        "text": "Film d'animation Pixar sur des jouets qui prennent vie quand les humains ne regardent pas. Woody, Buzz l'éclair, Andy, vers l'infini et au-delà, amitié entre jouets.",
        "category": "Animation",
    },
    {
        "title": "Le Roi Lion",
        "text": "Film d'animation Disney sur un lionceau qui doit reprendre sa place de roi. Simba, Hakuna Matata, Scar, Mufasa dans les nuages, cycle de la vie, savane africaine.",
        "category": "Animation",
    },
    {
        "title": "Vice-Versa",
        "text": "Film Pixar innovant qui se déroule dans la tête d'une jeune fille avec des émotions personnifiées. Joie, Tristesse, îlots de personnalité, souvenirs essentiels, train de la pensée.",
        "category": "Animation",
    },
    {
        "title": "Coco",
        "text": "Film Pixar sur un garçon mexicain qui voyage dans le monde des morts. Dia de los muertos, guitare, famille, Remember me, pétales de soucis, squelettes colorés.",
        "category": "Animation",
    },
    {
        "title": "Retour vers le Futur",
        "text": "Comédie de science-fiction sur un adolescent qui voyage dans le temps avec une DeLorean. 1.21 gigowatts, Doc Brown, Marty McFly, horloge, parents jeunes, skateboard.",
        "category": "Science-fiction",
    },
    {
        "title": "Star Wars",
        "text": "Saga spatiale épique entre rebelles et empire galactique. Lightsabers, Force, Darth Vader, Luke Skywalker, Death Star, Je suis ton père, X-Wing.",
        "category": "Science-fiction",
    },
    {
        "title": "E.T.",
        "text": "Film de Spielberg sur un extraterrestre perdu sur Terre qui se lie d'amitié avec un enfant. Vélo qui vole, doigt lumineux, téléphone maison, Reese's Pieces.",
        "category": "Science-fiction",
    },
]

WIKIPEDIA_DATA = [
    {
        "title": "Intelligence Artificielle",
        "text": "L'intelligence artificielle est un ensemble de théories et de techniques visant à réaliser des machines capables de simuler l'intelligence humaine. Les domaines incluent l'apprentissage automatique, le deep learning, les réseaux de neurones, le traitement du langage naturel. Applications dans la reconnaissance d'images, assistants vocaux, voitures autonomes, diagnostics médicaux.",
        "category": "Technologie",
    },
    {
        "title": "Machine Learning",
        "text": "L'apprentissage automatique est une branche de l'intelligence artificielle qui permet aux ordinateurs d'apprendre sans être explicitement programmés. Utilise des algorithmes statistiques pour identifier des patterns dans les données. Types principaux: supervisé, non-supervisé, par renforcement. Applications en prédiction, classification, clustering.",
        "category": "Technologie",
    },
    {
        "title": "Python",
        "text": "Python est un langage de programmation interprété, multi-paradigme et multiplateformes. Créé par Guido van Rossum en 1991. Syntaxe claire et lisible. Très utilisé en data science, intelligence artificielle, développement web, automatisation. Librairies populaires: NumPy, Pandas, TensorFlow, Django, Flask.",
        "category": "Technologie",
    },
    {
        "title": "Cryptomonnaie",
        "text": "Les cryptomonnaies sont des monnaies numériques utilisant la cryptographie pour sécuriser les transactions. Bitcoin créé en 2009 par Satoshi Nakamoto. Basé sur la blockchain, technologie de registre distribué. Ethereum, Solana, autres altcoins. Mining, proof of work, proof of stake, wallets, volatilité des cours.",
        "category": "Technologie",
    },
    {
        "title": "Seconde Guerre Mondiale",
        "text": "Conflit armé mondial de 1939 à 1945 opposant les Alliés aux puissances de l'Axe. Invasion de la Pologne, Pearl Harbor, débarquement de Normandie, bombes atomiques sur Hiroshima et Nagasaki. Plus de 70 millions de morts. Holocauste. Création de l'ONU après-guerre.",
        "category": "Histoire",
    },
    {
        "title": "Révolution Française",
        "text": "Période révolutionnaire en France de 1789 à 1799 qui transforme profondément la société. Prise de la Bastille, Déclaration des droits de l'homme, abolition de la monarchie, Terreur, guillotine, Napoléon Bonaparte. Liberté, égalité, fraternité. Fin de l'Ancien Régime.",
        "category": "Histoire",
    },
    {
        "title": "Jules César",
        "text": "Général et homme politique romain du premier siècle avant J.-C. Conquête de la Gaule, franchissement du Rubicon, guerre civile, dictateur perpétuel. Assassiné aux ides de mars en 44 av. J.-C. Et tu, Brute. Réformes du calendrier julien. Relation avec Cléopâtre.",
        "category": "Histoire",
    },
    {
        "title": "Napoléon Bonaparte",
        "text": "Empereur des Français de 1804 à 1815. Général brillant, campagnes d'Italie et d'Égypte, sacre à Notre-Dame, Code civil, guerres napoléoniennes, bataille d'Austerlitz, campagne de Russie, Waterloo, exil à Sainte-Hélène. A remodelé l'Europe.",
        "category": "Histoire",
    },
    {
        "title": "Coupe du Monde de Football",
        "text": "Compétition internationale de football organisée tous les quatre ans par la FIFA depuis 1930. Brésil pays le plus titré avec 5 victoires. France championne en 1998 et 2018. Finale spectaculaire, buts mémorables, penalties. Plus grand événement sportif au monde avec les Jeux Olympiques.",
        "category": "Sport",
    },
    {
        "title": "Jeux Olympiques",
        "text": "Compétition sportive internationale regroupant des sports d'été et d'hiver. Originaires de la Grèce antique, réintroduits en 1896 par Pierre de Coubertin. Flamme olympique, anneaux, cérémonies d'ouverture spectaculaires, records du monde, médailles d'or argent bronze.",
        "category": "Sport",
    },
    {
        "title": "Lionel Messi",
        "text": "Footballeur argentin considéré comme l'un des meilleurs joueurs de tous les temps. Carrière au FC Barcelone puis au PSG et Inter Miami. Sept Ballons d'Or, record de buts, dribbles magiques, pied gauche exceptionnel. Vainqueur de la Coupe du Monde 2022 avec l'Argentine.",
        "category": "Sport",
    },
    {
        "title": "Roger Federer",
        "text": "Joueur de tennis suisse légendaire, l'un des meilleurs de l'histoire. Vingt titres du Grand Chelem, jeu élégant et fluide, revers à une main, service précis. Rivalités avec Nadal et Djokovic. Fair-play et sportivité exemplaires.",
        "category": "Sport",
    },
    {
        "title": "Michael Jordan",
        "text": "Basketteur américain légendaire qui a dominé la NBA dans les années 90. Six championnats avec les Chicago Bulls, cinq MVP, dunks spectaculaires, langue tirée. Icône mondiale, Air Jordan, Space Jam. Considéré comme le plus grand joueur de basket de tous les temps.",
        "category": "Sport",
    },
    {
        "title": "Relativité Générale",
        "text": "Théorie de la gravitation élaborée par Albert Einstein entre 1907 et 1915. La gravité n'est pas une force mais une déformation de l'espace-temps causée par la masse. Prédictions: trous noirs, ondes gravitationnelles, expansion de l'univers. Confirmée par de nombreuses expériences.",
        "category": "Science",
    },
    {
        "title": "Mécanique Quantique",
        "text": "Branche de la physique qui étudie le comportement de la matière et de la lumière à l'échelle atomique. Principes d'incertitude de Heisenberg, dualité onde-corpuscule, superposition quantique, intrication. Applications en informatique quantique, lasers, transistors.",
        "category": "Science",
    },
    {
        "title": "ADN",
        "text": "Acide désoxyribonucléique, molécule support de l'information génétique. Structure en double hélice découverte par Watson et Crick en 1953. Composé de nucléotides: adénine, thymine, guanine, cytosine. Code génétique, réplication, mutations, séquençage du génome.",
        "category": "Science",
    },
    {
        "title": "Trou Noir",
        "text": "Région de l'espace-temps dont le champ gravitationnel est si intense que rien, pas même la lumière, ne peut s'en échapper. Formés par effondrement d'étoiles massives. Horizon des événements, singularité. Première photo en 2019 du trou noir M87. Trou noir supermassif au centre des galaxies.",
        "category": "Science",
    },
    {
        "title": "Photosynthèse",
        "text": "Processus biologique permettant aux plantes de produire de la matière organique à partir de lumière, eau et dioxyde de carbone. Chlorophylle, chloroplastes, oxygène produit. Base de la chaîne alimentaire. Cycle du carbone, lutte contre le réchauffement climatique.",
        "category": "Science",
    },
    {
        "title": "Mona Lisa",
        "text": "Tableau de Léonard de Vinci peint entre 1503 et 1519. Portrait de Lisa Gherardini, femme de Francesco del Giocondo. Sourire énigmatique, technique du sfumato, regard qui suit. Exposé au musée du Louvre à Paris. Oeuvre d'art la plus célèbre au monde.",
        "category": "Art",
    },
    {
        "title": "Van Gogh",
        "text": "Peintre néerlandais post-impressionniste du 19ème siècle. Style expressif avec coups de pinceau épais, couleurs vives. La Nuit étoilée, Les Tournesols, Autoportraits. Vie tourmentée, oreille coupée, troubles mentaux. Oeuvres peu reconnues de son vivant, désormais inestimables.",
        "category": "Art",
    },
    {
        "title": "Mozart",
        "text": "Compositeur autrichien du 18ème siècle, enfant prodige de la musique classique. Plus de 600 oeuvres: symphonies, concertos, opéras, musique de chambre. La Flûte enchantée, Requiem, Petite musique de nuit. Génie musical mort à 35 ans. Influence majeure sur la musique occidentale.",
        "category": "Art",
    },
    {
        "title": "Shakespeare",
        "text": "Dramaturge et poète anglais du 16-17ème siècle, considéré comme le plus grand écrivain de langue anglaise. Hamlet, Roméo et Juliette, Macbeth, Le Songe d'une nuit d'été. Exploration de la nature humaine, tragédies, comédies. Théâtre du Globe à Londres. Citations universelles.",
        "category": "Art",
    },
    {
        "title": "Réchauffement Climatique",
        "text": "Augmentation de la température moyenne de la Terre causée principalement par les émissions de gaz à effet de serre d'origine humaine. Fonte des glaces, montée des océans, événements climatiques extrêmes. Accord de Paris, objectif 1.5°C, énergies renouvelables, réduction des émissions de CO2.",
        "category": "Environnement",
    },
    {
        "title": "Biodiversité",
        "text": "Variété des formes de vie sur Terre: plantes, animaux, micro-organismes, écosystèmes. Sixième extinction de masse en cours causée par l'activité humaine. Déforestation, surpêche, pollution, changement climatique. Importance pour l'équilibre des écosystèmes, services écosystémiques, préservation des espèces.",
        "category": "Environnement",
    },
    {
        "title": "Pyramides d'Égypte",
        "text": "Monuments funéraires construits sous l'Ancien Empire égyptien. Pyramide de Khéops à Gizeh, seule des sept merveilles du monde antique encore debout. Construction avec des millions de blocs de pierre. Tombeau des pharaons, momification, trésors. Sphinx, hiéroglyphes, archéologie.",
        "category": "Architecture",
    },
    {
        "title": "Tour Eiffel",
        "text": "Monument parisien emblématique construit par Gustave Eiffel pour l'Exposition universelle de 1889. Structure métallique de 330 mètres de hauteur. Critiquée à sa construction, devenue symbole de Paris et de la France. Illuminations nocturnes, restaurants, millions de visiteurs chaque année.",
        "category": "Architecture",
    },
]

LIVRES_DATA = [
    {
        "title": "Les Misérables - Victor Hugo (1802-1885)",
        "text": "Roman social français publié en 1862. L'histoire de Jean Valjean, ancien forçat devenu maire, poursuivi par l'inspecteur Javert. Fresque historique de la France du 19e siècle avec Cosette, les Thénardier, Marius. Thèmes de la rédemption, de la justice sociale et de l'amour. Chef-d'œuvre du romantisme.",
        "category": "XIXe siècle",
    },
    {
        "title": "Le Comte de Monte-Cristo - Alexandre Dumas (1802-1870)",
        "text": "Roman d'aventures de 1844. Edmond Dantès, injustement emprisonné au château d'If, s'évade après 14 ans et devient richissime grâce au trésor de l'abbé Faria. Il orchestre sa vengeance méthodique contre Fernand, Danglars et Villefort qui l'ont trahi. Haletant et épique.",
        "category": "XIXe siècle",
    },
    {
        "title": "L'Étranger - Albert Camus (1913-1960)",
        "text": "Roman publié en 1942. Meursault, un homme indifférent aux conventions sociales, tue un Arabe sur une plage algérienne. Procès où il est jugé plus pour son absence d'émotion à l'enterrement de sa mère que pour son crime. Philosophie de l'absurde et de l'aliénation.",
        "category": "XXe siècle",
    },
    {
        "title": "Le Petit Prince - Antoine de Saint-Exupéry (1900-1944)",
        "text": "Conte philosophique publié en 1943. Un aviateur en panne dans le désert rencontre un petit prince venu de l'astéroïde B-612. Réflexions poétiques sur l'amitié, l'amour, la perte de l'innocence. Le renard et la rose. 'On ne voit bien qu'avec le cœur. L'essentiel est invisible pour les yeux.'",
        "category": "XXe siècle",
    },
    {
        "title": "Madame Bovary - Gustave Flaubert (1821-1880)",
        "text": "Roman réaliste de 1857. Emma Bovary, épouse d'un médecin de province, s'ennuie dans sa vie bourgeoise et cherche l'amour passionnel dans des liaisons adultères avec Rodolphe et Léon. Sa quête romantique la mène à la ruine financière et au suicide. Critique du romantisme.",
        "category": "XIXe siècle",
    },
    {
        "title": "Germinal - Émile Zola (1840-1902)",
        "text": "Roman naturaliste de 1885, partie du cycle des Rougon-Macquart. Étienne Lantier arrive à la mine de Montsou et découvre les conditions misérables des mineurs. Grève, révolte ouvrière, répression sanglante. Peinture sombre de la condition ouvrière au 19e siècle. Engagement social de Zola.",
        "category": "XIXe siècle",
    },
    {
        "title": "Candide - Voltaire (1694-1778)",
        "text": "Conte philosophique satirique de 1759. Candide, jeune homme naïf qui croit à l'optimisme de son maître Pangloss ('tout est pour le mieux dans le meilleur des mondes possibles'), traverse guerres, tremblements de terre, Inquisition. Critique féroce de l'optimisme philosophique et des institutions.",
        "category": "XVIIIe siècle",
    },
    {
        "title": "Notre-Dame de Paris - Victor Hugo (1802-1885)",
        "text": "Roman gothique de 1831. Quasimodo, le sonneur de cloches bossu de Notre-Dame, aime Esmeralda, bohémienne poursuivie par l'archidiacre Frollo. Paris médiéval, architecture gothique, amour impossible, drame romantique. Sauvegarde du patrimoine architectural.",
        "category": "XIXe siècle",
    },
]


def _load_hardcoded(
    name: str, extended: bool, sample_size: Optional[int]
) -> List[Dict[str, str]]:
    """
    Charge les PETITS datasets hardcodés (fallback d'urgence uniquement!)
    Ne devrait être utilisé QUE si les fichiers synthetic/ ou HuggingFace échouent.
    """
    base_datasets = {
        "recettes": RECETTES_DATA,  # 30 recettes
        "films": FILMS_DATA,  # 37 films
        "wikipedia": WIKIPEDIA_DATA,  # 27 articles
    }

    if name not in base_datasets:
        raise ValueError(
            f"Dataset '{name}' inconnu. Choisir parmi: {list(base_datasets.keys())}"
        )

    dataset = base_datasets[name]

    # Si sample_size spécifié, limiter
    if sample_size and sample_size < len(dataset):
        dataset = dataset[:sample_size]

    print(f"⚠️ FALLBACK HARDCODÉ: {len(dataset)} documents (données minimales)")
    return dataset


# === SUPPRESSION DES ANCIENNES FONCTIONS DE GÉNÉRATION ===
# Les fonctions _generate_extended_* ont été supprimées car obsolètes.
# On utilise maintenant:
# - data/synthetic/*.json pour recettes/films
# - Git pour wikipedia (data/datasets/wikipedia_fr.json)


def _load_from_local_json(name: str) -> Optional[List[Dict[str, str]]]:
    """
    Charge un dataset depuis les fichiers JSON locaux (versionnés dans git)

    Args:
        name: 'wikipedia' uniquement

    Returns:
        Liste de documents ou None si fichier absent
    """
    if name != "wikipedia":
        return None

    filepath = DATASETS_DIR / "wikipedia_fr.json"

    if not filepath.exists():
        print(f"⚠️ Fichier {filepath.name} non trouvé!")
        print(f"   Normalement versionné dans git, vérifie ton clone!")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Chargé depuis git: {filepath.name} ({len(data)} documents)")
        return data
    except Exception as e:
        print(f"❌ Erreur lecture {filepath.name}: {e}")
        return None


def load_dataset(
    name: str = "recettes",
    use_cache: bool = True,
    sample_size: Optional[int] = None,
    extended: bool = False,
) -> List[Dict[str, str]]:
    """
    Charge un dataset:
    1. Mode normal: Depuis data/synthetic/ (recettes/films) ou hardcodé (fallback)
    2. Mode étendu: Depuis data/datasets/ (JSON pré-téléchargés) ou HuggingFace (fallback)

    Args:
        name: 'recettes', 'films', ou 'wikipedia'
        use_cache: Utiliser le cache si disponible (legacy, ignoré)
        sample_size: Nombre de docs à charger (None = défaut selon mode)
        extended: Si True, charge la version étendue

    Returns:
        List[Dict]: [{'title': str, 'text': str, 'category': str}, ...]
    """
    # === MODE ÉTENDU ===
    if extended:
        print(f"🌐 Mode étendu: {name}...")

        # Wikipedia: charger depuis fichier JSON local (versionné dans git)
        if name == "wikipedia":
            local_data = _load_from_local_json(name)

            if local_data:
                # Limiter si sample_size spécifié
                if sample_size and sample_size < len(local_data):
                    print(f"✂️ Limité à {sample_size} documents")
                    return local_data[:sample_size]
                return local_data
            else:
                print(f"⚠️ Fallback: tentative chargement depuis HuggingFace...")
                # Fallback: essayer HuggingFace (méthode legacy, lente)
                if HF_AVAILABLE:
                    target_size = sample_size if sample_size else 1000
                    return _load_wikipedia_hf(
                        target_size=target_size, use_cache=use_cache
                    )

        # Recettes/Films: charger depuis synthetic (TOUTES les données)
        elif name in ["recettes", "films"]:
            return _load_from_synthetic(name, target_size=None)
        else:
            print(f"⚠️ Dataset '{name}' inconnu, fallback hardcodé...")
            return _load_hardcoded(name, extended=False, sample_size=sample_size)

    # === MODE NORMAL: SYNTHÉTIQUE OU HUGGING FACE (200 docs) ===
    if name in ["recettes", "films"]:
        # Charger depuis data/synthetic/ (50 docs par défaut)
        target_size = sample_size if sample_size else 50
        return _load_from_synthetic(name, target_size=target_size)

    elif name == "wikipedia" and HF_AVAILABLE:
        # Charger depuis HuggingFace avec limite de 200 docs
        target_size = sample_size if sample_size else 200
        return _load_wikipedia_hf(target_size=target_size, use_cache=use_cache)

    else:
        # Fallback hardcodé (dernière chance)
        print(f"📦 Chargement de '{name}' hardcodé...")
        return _load_hardcoded(name, extended=False, sample_size=sample_size)


def _load_from_synthetic(name: str, target_size: int = 50) -> List[Dict[str, str]]:
    """
    Charge un dataset depuis les fichiers synthétiques dans data/synthetic/

    Args:
        name: 'recettes' ou 'films'
        target_size: Nombre de docs à charger (None = tous)

    Returns:
        List[Dict]: Liste de documents
    """
    # Mapping nom → fichier
    file_mapping = {"recettes": "recipes_fr.json", "films": "films_fr.json"}

    if name not in file_mapping:
        raise ValueError(f"Dataset synthétique '{name}' non disponible!")

    file_path = Path(__file__).parent.parent / "data" / "synthetic" / file_mapping[name]

    try:
        print(f"📦 Chargement de '{name}' depuis {file_path.name}...")

        # Lire le fichier JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_available = len(data)
        print(f"📊 {total_available} documents disponibles dans le fichier")

        # Si target_size spécifié ET inférieur au total, limiter
        if target_size is not None and target_size < total_available:
            data = data[:target_size]
            print(f"✂️ Limité à {target_size} documents")
        else:
            print(f"📖 Chargement de TOUS les documents ({total_available})")

        # Convertir au format attendu (title, text, category)
        documents = []
        for item in data:
            documents.append(
                {
                    "title": item.get("title", "Sans titre"),
                    "text": item.get("text", ""),
                    "category": item.get("category", "Divers"),
                }
            )

        print(f"✅ {len(documents)} documents chargés depuis {file_path.name}")
        return documents

    except FileNotFoundError:
        print(f"❌ Fichier {file_path} non trouvé!")
        print(f"   Fallback sur données hardcodées...")

        # Fallback: utiliser les données hardcodées
        if name == "recettes":
            return RECETTES_DATA[:target_size] if target_size else RECETTES_DATA
        elif name == "films":
            return FILMS_DATA[:target_size] if target_size else FILMS_DATA
        else:
            return []

    except json.JSONDecodeError as e:
        print(f"❌ Erreur de parsing JSON: {e}")
        print(f"   Fallback sur données hardcodées...")

        if name == "recettes":
            return RECETTES_DATA[:target_size] if target_size else RECETTES_DATA
        elif name == "films":
            return FILMS_DATA[:target_size] if target_size else FILMS_DATA
        else:
            return []

    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        print(f"   Fallback sur données hardcodées...")

        if name == "recettes":
            return RECETTES_DATA[:target_size] if target_size else RECETTES_DATA
        elif name == "films":
            return FILMS_DATA[:target_size] if target_size else FILMS_DATA
        else:
            return []


def _load_wikipedia_hf(
    target_size: int = 1000, use_cache: bool = True
) -> List[Dict[str, str]]:
    """
    Charge de VRAIS articles Wikipedia FR depuis Hugging Face

    Args:
        target_size: Nombre d'articles à charger (1000 ou 10000)
        use_cache: Utiliser le cache pour éviter retéléchargements

    Returns:
        List[Dict]: Articles Wikipedia avec title, text, category
    """
    if not HF_AVAILABLE:
        print("❌ Hugging Face datasets non disponible!")
        return []

    # Vérifier le cache
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"wikipedia_{target_size}.pkl"

    if use_cache and cache_file.exists():
        print(f"📦 Chargement depuis le cache: {cache_file.name}")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Erreur lecture cache: {e}")
            # Continue pour retélécharger

    print(f"🌐 Téléchargement de {target_size} articles Wikipedia FR...")
    print("⏳ Cela peut prendre quelques minutes la première fois...")

    try:
        # Charger Wikipedia FR en streaming (pas tout télécharger!)
        wiki = hf_load_dataset(
            "wikimedia/wikipedia",
            "20231101.fr",
            split="train",
            streaming=True,  # ← CRUCIAL: évite de télécharger les 50GB!
        )

        # Shuffle pour avoir de la DIVERSITÉ (pas triés par sujet!)
        wiki_shuffled = wiki.shuffle(seed=42, buffer_size=10000)

        # Collecter les articles (seulement title et text!)
        articles = []
        for i, item in enumerate(wiki_shuffled):
            if len(articles) >= target_size:
                break

            # Extraire seulement ce qu'on veut
            title = item.get("title", "Sans titre")
            text = item.get("text", "")

            # Filtrer les articles trop courts ou vides
            if len(text.strip()) < 100:  # Au moins 100 caractères
                continue

            # Limiter la longueur du texte (garder premiers 2000 caractères)
            if len(text) > 2000:
                text = text[:2000] + "..."

            # Essayer de deviner la catégorie depuis le titre/contenu
            category = _guess_wikipedia_category(title, text)

            articles.append({"title": title, "text": text, "category": category})

            # Progress indicator
            if (len(articles) % 100) == 0:
                print(f"   ... {len(articles)}/{target_size} articles chargés")

        print(f"✅ {len(articles)} articles Wikipedia chargés avec succès!")

        # Sauvegarder dans le cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(articles, f)
            print(f"💾 Cache sauvegardé: {cache_file.name}")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde cache: {e}")

        return articles

    except Exception as e:
        print(f"❌ Erreur chargement Wikipedia: {e}")
        print(f"   Type d'erreur: {type(e).__name__}")
        print("   Fallback sur données hardcodées...")
        return _generate_extended_wikipedia()


def _load_recettes_hf(
    target_size: int = 1000, use_cache: bool = True
) -> List[Dict[str, str]]:
    """
    Charge de VRAIES recettes depuis les fichiers synthétiques (data/synthetic/)
    En mode étendu, on charge TOUTES les recettes disponibles

    Args:
        target_size: Nombre de recettes à charger
        use_cache: Utiliser le cache si disponible (ignoré pour synthetic)

    Returns:
        List[Dict]: Liste de recettes
    """
    print(
        f"📥 Chargement de TOUTES les recettes depuis data/synthetic/recipes_fr.json..."
    )

    # Charger TOUTES les recettes (pas de limite!)
    return _load_from_synthetic("recettes", target_size=None)  # None = TOUT!


def _load_films_hf(
    target_size: int = 1000, use_cache: bool = True
) -> List[Dict[str, str]]:
    """
    Charge de VRAIS films depuis les fichiers synthétiques (data/synthetic/)
    En mode étendu, on charge TOUS les films disponibles

    Args:
        target_size: Nombre de films à charger
        use_cache: Utiliser le cache si disponible (ignoré pour synthetic)

    Returns:
        List[Dict]: Liste de films
    """
    print(f"📥 Chargement de TOUS les films depuis data/synthetic/films_fr.json...")

    # Charger TOUS les films (pas de limite!)
    return _load_from_synthetic("films", target_size=None)  # None = TOUT!


# Fonctions liées aux livres supprimées (dataset trop lourd: 400 MB)

        "À propos de ",
        "Concernant ",
    ]
    suffixes = [
        "",
        " Plus de détails disponibles.",
        " Article détaillé.",
        " Informations complémentaires.",
        " Source fiable.",
    ]

    while len(result) < target_size:
        # Prendre un doc au hasard
        base_doc = random.choice(base_docs)

        # Créer une variation
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        variation = {
            "title": f"{base_doc['title']} #{counter}",
            "text": f"{prefix}{base_doc['text']}{suffix}",
            "category": base_doc["category"],
        }

        result.append(variation)
        counter += 1

    return result[:target_size]


def _load_hardcoded(
    name: str, extended: bool, sample_size: Optional[int]
) -> List[Dict[str, str]]:
    """Charge les datasets hardcodés (fallback)"""
    # Déterminer la taille cible
    if sample_size:
        target_size = sample_size
    elif extended:
        target_size = 10000  # Extended = 10k docs
    else:
        target_size = 1000  # Normal = 1k docs

    # Charger les données de base
    if extended or target_size > 300:
        # Utiliser les versions étendues comme base
        base_datasets = {
            "recettes": _generate_extended_recettes(),
            "films": _generate_extended_films(),
            "livres": LIVRES_DATA,  # Pas de version étendue hardcodée pour livres
            "wikipedia": _generate_extended_wikipedia(),
        }
    else:
        # Utiliser les versions normales comme base
        base_datasets = {
            "recettes": RECETTES_DATA,
            "films": FILMS_DATA,
            "livres": LIVRES_DATA,
            "wikipedia": WIKIPEDIA_DATA,
        }

    if name not in base_datasets:
        raise ValueError(
            f"Dataset '{name}' inconnu. Choisir parmi: {list(base_datasets.keys())}"
        )

    base_docs = base_datasets[name]

    # Multiplier pour atteindre la taille cible
    dataset = _multiply_dataset(base_docs, target_size)

    print(
        f"📊 Dataset '{name}' chargé: {len(dataset)} documents (target: {target_size})"
    )

    return dataset


def _load_from_huggingface(
    name: str, extended: bool, use_cache: bool, sample_size: Optional[int]
) -> List[Dict[str, str]]:
    """
    Charge un dataset depuis Hugging Face
    1k docs (normal) ou 10k docs (extended)
    """
    # Configuration des datasets HF
    hf_configs = {
        "recettes": {
            "path": "opus_books",  # Corpus de livres (on utilise comme proxy)
            "config": "fr-en",
            "text_col": "translation",  # On prendra le côté FR
            "title_col": None,
            "category_col": None,
            "use_translation_fr": True,  # Flag pour extraire le français
        },
        "films": {
            "path": "allocine",
            "config": None,
            "text_col": "review",
            "title_col": None,
            "category_col": "polarity",
        },
        "wikipedia": {
            "path": "wikipedia",
            "config": "20220301.fr",
            "text_col": "text",
            "title_col": "title",
            "category_col": None,
        },
    }

    if name not in hf_configs:
        raise ValueError(f"Dataset HF '{name}' non configuré")

    config = hf_configs[name]

    # Déterminer la taille cible
    if sample_size:
        target_size = sample_size
    else:
        target_size = 10000 if extended else 1000

    # Vérifier le cache
    cache_dir = Path("data/cache_hf")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}_{'ext' if extended else 'norm'}_{target_size}.pkl"

    if use_cache and cache_file.exists():
        print(f"📦 Cache HF trouvé: {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"📥 Téléchargement HF: {name} ({target_size} docs)...")

    # Charger depuis HF
    if name == "wikipedia":
        # Wikipedia nécessite streaming (trop gros!)
        ds = hf_load_dataset(
            config["path"],
            config["config"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        documents = []
        for i, item in enumerate(ds):
            if len(documents) >= target_size:
                break

            text = item.get(config["text_col"], "")
            title = item.get(config["title_col"], f"Article {i + 1}")

            # Filtrer docs trop courts/longs
            if 200 < len(text) < 3000:
                documents.append(
                    {"title": title[:100], "text": text[:2000], "category": "Wikipedia"}
                )

    else:
        # Autres datasets: chargement complet
        try:
            ds = hf_load_dataset(config["path"], split="train", trust_remote_code=True)
        except:
            ds = hf_load_dataset(
                config["path"], split="train[:10000]", trust_remote_code=True
            )

        total = len(ds)
        n_samples = min(target_size, total)
        indices = random.sample(range(total), n_samples)

        documents = []
        for idx in indices:
            item = ds[idx]

            text = item.get(config["text_col"], "")
            title = (
                item.get(config["title_col"])
                if config["title_col"]
                else f"{name.title()} #{idx}"
            )
            category = item.get(config["category_col"], name.capitalize())

            if isinstance(text, str) and len(text) > 100:
                documents.append(
                    {
                        "title": str(title)[:100],
                        "text": str(text)[:2000],
                        "category": str(category),
                    }
                )

    print(f"✅ {len(documents)} documents HF chargés!")

    # Sauvegarder dans le cache
    with open(cache_file, "wb") as f:
        pickle.dump(documents, f)
    print(f"💾 Mis en cache: {cache_file.name}")

    return documents


def get_dataset_info(name: str, extended: bool = False) -> Dict:
    """
    Retourne des informations sur un dataset

    Args:
        name: Nom du dataset
        extended: Si True, charge extended version

    Returns:
        Dict avec infos: nb_docs, categories, description
    """
    dataset = load_dataset(name, extended=extended)
    categories = list(set(doc["category"] for doc in dataset))

    # Descriptions mises à jour
    # Vérifier si le fichier Wikipedia JSON local existe
    wiki_local = (DATASETS_DIR / "wikipedia_fr.json").exists()

    descriptions = {
        "recettes": f"Recettes françaises {'(synthetic TOUTES ~1200 docs 🍝)' if extended else '(synthetic 50 docs 🍝)'}",
        "films": f"Films français {'(synthetic TOUS ~1200 docs 🎬)' if extended else '(synthetic 50 docs 🎬)'}",
        "wikipedia": f"Articles Wikipedia FR {'(git 1000 docs 📚)' if (wiki_local and extended) else '(hardcodé 27 docs 📚)'}",
    }

    return {
        "name": name,
        "nb_docs": len(dataset),
        "categories": sorted(categories),
        "description": descriptions.get(name, "Dataset"),
    }


def get_all_datasets_info() -> List[Dict]:
    """
    Retourne les infos de tous les datasets disponibles
    """
    return [
        get_dataset_info(name) for name in ["recettes", "films", "wikipedia"]
    ]
