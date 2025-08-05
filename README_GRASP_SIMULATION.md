# Simulation de Grasping G1 Robot

Ce projet contient des scripts pour créer des vidéos de simulation de la phase de grasping du robot G1, avec détection de contact via les force sensors et fermeture des doigts sur le cube.

## Fonctionnalités

- **Détection de contact** : Utilisation des force sensors pour détecter le contact avec le cube
- **Grasping automatique** : Fermeture des doigts une fois le contact détecté
- **Système de récompenses** : Récompenses et pénalités basées sur le succès du grasping
- **Enregistrement vidéo** : Génération automatique de vidéos de simulation
- **Modèle amélioré** : Option pour ajouter des force sensors au modèle G1

## Fichiers inclus

### Scripts principaux
- `run_grasp_simulation.py` : Script principal pour lancer les simulations
- `grasp_simulation.py` : Simulation basique de grasping
- `grasp_simulation_advanced.py` : Simulation avancée avec stabilité
- `add_force_sensors.py` : Ajout de force sensors au modèle G1

### Modèles
- `results/g1_combined.xml` : Modèle G1 de base
- `results/g1_combined_with_force_sensors.xml` : Modèle avec force sensors (généré automatiquement)

## Installation

1. **Dépendances requises** :
```bash
pip install mujoco mujoco-viewer opencv-python numpy
```

2. **Vérification du modèle** :
Assurez-vous que le fichier `results/g1_combined.xml` existe dans votre workspace.

## Utilisation

### Simulation basique

```bash
# Simulation simple avec le modèle de base
python run_grasp_simulation.py

# Simulation avec paramètres personnalisés
python run_grasp_simulation.py --steps 1500 --contact-threshold 0.1
```

### Simulation avancée

```bash
# Simulation avancée avec vérification de stabilité
python run_grasp_simulation.py --advanced

# Simulation avancée avec modèle amélioré
python run_grasp_simulation.py --advanced --enhanced
```

### Options disponibles

- `--model` : Chemin vers le modèle MuJoCo (défaut: `results/g1_combined.xml`)
- `--enhanced` : Utiliser le modèle avec force sensors améliorés
- `--advanced` : Utiliser la simulation avancée avec stabilité
- `--no-video` : Ne pas sauvegarder de vidéo
- `--video-path` : Chemin de sortie pour la vidéo (défaut: `grasp_simulation.mp4`)
- `--steps` : Nombre maximum d'étapes (défaut: 2000)
- `--contact-threshold` : Seuil de détection de contact (défaut: 0.05)

## Exemples d'utilisation

### 1. Simulation rapide sans vidéo
```bash
python run_grasp_simulation.py --no-video --steps 500
```

### 2. Simulation complète avec vidéo haute qualité
```bash
python run_grasp_simulation.py --advanced --enhanced --steps 3000 --video-path "grasp_demo.mp4"
```

### 3. Test de différents seuils de contact
```bash
python run_grasp_simulation.py --contact-threshold 0.02
python run_grasp_simulation.py --contact-threshold 0.1
```

## Système de récompenses

### Récompenses positives
- **Contact détecté** : +2.0 points
- **Grasping réussi** : +10.0 points
- **Grasping stable** : +20.0 points (simulation avancée)
- **Hauteur du cube** : +5.0 × hauteur (plus le cube est haut, mieux c'est)

### Pénalités
- **Mouvements excessifs** : -15.0 × distance (si le cube s'éloigne trop)
- **Temps** : -0.1 par step après 1000 steps

## Détection de contact

### Force Sensors
Le système utilise les force sensors intégrés dans le modèle G1 pour détecter le contact :
- `left_thumb_force_sensor_*`
- `left_index_force_sensor_*`
- `left_middle_force_sensor_*`
- `left_ring_force_sensor_*`
- `right_thumb_force_sensor_*`
- `right_index_force_sensor_*`
- `right_middle_force_sensor_*`
- `right_ring_force_sensor_*`

### Seuil de détection
Le seuil par défaut est de 0.05, mais peut être ajusté avec `--contact-threshold`.

## Logique de grasping

1. **Phase initiale** : Les doigts sont ouverts, le robot attend le contact
2. **Détection de contact** : Une fois le contact détecté, les doigts commencent à se fermer
3. **Fermeture progressive** : Les doigts se ferment progressivement jusqu'à la position fermée
4. **Vérification de stabilité** : (simulation avancée) Vérification que le cube est stable
5. **Grasping complet** : Le grasping est considéré comme réussi

## Sorties

### Vidéo
- Format : MP4
- Fréquence : 30 FPS
- Résolution : Dépend de la fenêtre de visualisation

### Console
- Progression en temps réel
- Informations sur les sensors
- Résultats finaux détaillés

### Métriques
- Récompense totale
- Nombre d'étapes
- État du contact et du grasping
- Position finale du cube

## Dépannage

### Erreurs communes

1. **Modèle non trouvé** :
   ```
   Erreur: Le modèle results/g1_combined.xml n'existe pas!
   ```
   Solution : Vérifiez que le fichier existe dans le bon répertoire.

2. **Erreur d'import** :
   ```
   ImportError: No module named 'mujoco'
   ```
   Solution : Installez les dépendances avec `pip install mujoco mujoco-viewer opencv-python numpy`

3. **Force sensors non trouvés** :
   ```
   Force sensors non trouvés, utilisation des sensors de position des doigts
   ```
   Solution : Utilisez l'option `--enhanced` pour ajouter des force sensors.

### Optimisation des performances

- Réduisez le nombre d'étapes avec `--steps` pour des tests rapides
- Utilisez `--no-video` pour éviter l'overhead de l'enregistrement vidéo
- Ajustez le seuil de contact selon vos besoins

## Personnalisation

### Ajout de nouveaux sensors
Modifiez `add_force_sensors.py` pour ajouter de nouveaux types de sensors.

### Modification des récompenses
Modifiez la méthode `_compute_reward()` dans les classes de simulation.

### Changement de la logique de grasping
Modifiez la méthode `step()` pour implémenter de nouvelles stratégies.

## Support

Pour toute question ou problème, vérifiez :
1. Que toutes les dépendances sont installées
2. Que le modèle G1 existe et est valide
3. Les logs de la console pour les messages d'erreur détaillés