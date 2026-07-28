# Évaluation du modèle

Ce module compare les prédictions du bot à des observations réelles passées, pour produire des métriques de performance internes.

## Pourquoi

Avant toute évolution de l'algorithme, on a besoin d'une mesure objective de la qualité actuelle du modèle, pour savoir si un changement l'améliore ou le dégrade.

## Principe général

Les prédictions sont rejouées sur des dates passées, en s'assurant qu'aucune information future n'est utilisée (l'évaluation ne doit jamais "tricher" en connaissant par avance ce qui s'est réellement passé).

Les métriques précises et leurs seuils d'interprétation ne sont pas détaillés publiquement.

## Stockage

Les résultats sont archivés séparément des autres données, mis à jour automatiquement.
