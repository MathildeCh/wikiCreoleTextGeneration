# Génération Automatique de Texte en WikiCréole

## Introduction

Ce projet vise à générer automatiquement du texte en syntaxe WikiCréole en utilisant un réseau de neurones récurrents (RNN) avec TensorFlow. Le but est de produire une page Wikipédia sur une nouvelle région française, en s'inspirant des pages existantes des régions françaises.

## Objectifs

1. **Prétraitement des données** : Convertir le texte WikiCréole en séquences de caractères.
2. **Entraînement d'un modèle RNN** : Utiliser un modèle de réseau récurrent pour prédire le prochain caractère dans une séquence.
3. **Génération de texte** : Créer une nouvelle page Wikipédia en WikiCréole pour une région fictive.

## Méthodologie

1. **Importation des Librairies** : TensorFlow, NumPy, et autres.
2. **Chargement du Corpus** : Lire les fichiers WikiCréole des pages des régions françaises.
3. **Prétraitement** : Conversion des caractères en identifiants, génération des séquences de caractères.
4. **Entraînement du Modèle** : Utilisation d'un modèle RNN avec des couches d'Embedding, GRU et Dense, entraîné sur 100 époques.
5. **Génération de Texte** : Utilisation du modèle entraîné pour générer du texte en WikiCréole à partir d'une graine textuelle.

## Résultats

- **Textes générés** : Exemples de nouvelles pages Wikipédia sur des régions fictives de France écrites en WikiCréole.

## Contributeurs

- **Diego Rossini** - [GitLab](https://gitlab.com/diego.rossini418)
- **Mathilde Charlet** - [GitLab](https://gitlab.com/mathildecharletb)
