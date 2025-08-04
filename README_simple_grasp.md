# Système de Grasping Simplifié

Ce projet implémente un système d'apprentissage par renforcement pour l'apprentissage du grasping avec un robot. Le système utilise l'algorithme SAC (Soft Actor-Critic) et MuJoCo pour la simulation.

## 🎯 Objectif

Le robot doit apprendre à :
1. **Détecter un cube** dans l'environnement
2. **Établir le contact** avec le cube
3. **Fermer ses doigts** pour saisir le cube
4. **Soulever le cube** pour valider le grasping

## 📁 Structure du Projet

```
├── agents/
│   └── simple_sac_agent.py      # Agent SAC simplifié
├── envs/
│   └── simple_grasp_env.py      # Environnement de simulation
├── tasks/
│   └── grasp/
│       └── simple_grasp_task.py # Tâche de grasping
├── config/
│   └── simple_grasp_config.yaml # Configuration
├── train_simple_grasp.py        # Script d'entraînement
├── test_simple_grasp.py         # Script de test
└── README_simple_grasp.md       # Ce fichier
```

## 🚀 Installation et Utilisation

### 1. Prérequis

```bash
# Installer les dépendances
pip install torch numpy mujoco gymnasium opencv-python matplotlib pyyaml
```

### 2. Test du Système

Avant de lancer l'entraînement, testez que tout fonctionne :

```bash
python test_simple_grasp.py
```

Ce script teste :
- ✅ L'environnement de simulation
- ✅ L'agent SAC
- ✅ La tâche de grasping
- ✅ L'intégration complète

### 3. Entraînement

#### Entraînement rapide (100 épisodes)
```bash
python train_simple_grasp.py --episodes 100
```

#### Entraînement complet (1000 épisodes)
```bash
python train_simple_grasp.py --episodes 1000
```

#### Entraînement avec configuration personnalisée
```bash
python train_simple_grasp.py --config config/simple_grasp_config.yaml
```

### 4. Résultats

L'entraînement génère automatiquement :

- 📁 **`results/videos/`** : Vidéos des épisodes d'entraînement
- 📁 **`results/models/`** : Modèles sauvegardés
- 📁 **`results/logs/`** : Graphiques de progression

## 🔧 Configuration

Le fichier `config/simple_grasp_config.yaml` permet de personnaliser :

### Environnement
```yaml
env:
  xml_path: "assets/scenes/complete_scene.xml"  # Modèle MuJoCo
  max_steps_per_episode: 1000                   # Étapes max par épisode
  touch_sensors: ["touch1_sensor", "touch2_sensor"]  # Capteurs tactiles
```

### Agent SAC
```yaml
agent:
  hidden_sizes: [256, 256]  # Architecture du réseau
  lr: 3e-4                  # Taux d'apprentissage
  gamma: 0.99               # Facteur de discount
  alpha: 0.2                # Paramètre d'entropie
```

### Entraînement
```yaml
training:
  max_episodes: 1000        # Nombre d'épisodes
  batch_size: 256           # Taille du batch
  save_frequency: 100       # Sauvegarde tous les 100 épisodes
  eval_frequency: 50        # Vidéo tous les 50 épisodes
```

## 🎮 Fonctionnalités

### ✅ Fonctionnalités Implémentées

1. **Environnement de Simulation**
   - Modèle MuJoCo simple avec robot et cube
   - Capteurs tactiles pour détecter le contact
   - Rendu vidéo automatique

2. **Agent SAC**
   - Réseaux de neurones pour acteur et critiques
   - Buffer de replay pour l'expérience
   - Mise à jour des réseaux cibles

3. **Tâche de Grasping**
   - Détection de contact
   - Récompenses pour le grasping
   - Suivi de l'état de la tâche

4. **Entraînement**
   - Sauvegarde automatique des modèles
   - Génération de vidéos de simulation
   - Graphiques de progression
   - Statistiques en temps réel

### 🎯 Système de Récompenses

- **Contact détecté** : +10 points
- **Grasping réussi** : +50 points
- **Soulever le cube** : +1 point par unité de hauteur
- **Pénalité d'action** : -0.01 × moyenne des actions²

## 📊 Monitoring

Pendant l'entraînement, vous verrez :

```
Episode 10/1000 - Reward: 15.23 (avg: 12.45) - Length: 156 - Success Rate: 20.00%
Episode 20/1000 - Reward: 23.67 (avg: 18.34) - Length: 142 - Success Rate: 25.00%
...
```

## 🎥 Vidéos de Simulation

Les vidéos sont automatiquement générées et sauvegardées dans `results/videos/` :
- Format : MP4
- Fréquence : Tous les 50 épisodes (configurable)
- Résolution : 640x480 (configurable)

## 📈 Graphiques

Après l'entraînement, des graphiques sont générés dans `results/logs/` :
- Récompenses par épisode
- Récompense moyenne mobile
- Longueur des épisodes
- Taux de succès cumulatif

## 🔍 Dépannage

### Erreurs Courantes

1. **"Capteur tactile non trouvé"**
   - Le modèle MuJoCo n'a pas les capteurs spécifiés
   - Un modèle simple sera créé automatiquement

2. **"Cube non trouvé"**
   - Le modèle MuJoCo n'a pas de cube
   - Un cube sera ajouté automatiquement

3. **Erreur de rendu**
   - MuJoCo viewer non disponible
   - Le rendu vidéo sera désactivé

### Solutions

- Vérifiez que MuJoCo est installé correctement
- Utilisez le script de test pour diagnostiquer les problèmes
- Consultez les logs d'erreur pour plus de détails

## 🚀 Prochaines Étapes

Pour améliorer le système :

1. **Modèle plus complexe** : Ajouter un robot plus réaliste
2. **Capteurs avancés** : Caméras, capteurs de force
3. **Tâches multiples** : Pick & place, manipulation d'objets
4. **Environnements variés** : Différents objets, obstacles
5. **Algorithmes avancés** : PPO, TD3, autres algorithmes RL

## 📝 Notes Techniques

- **Device** : CPU par défaut, GPU automatique si disponible
- **Précision** : Float32 pour les observations et actions
- **Mémoire** : Buffer de replay de 100k expériences
- **Performance** : Optimisé pour l'entraînement rapide

## 🤝 Contribution

Pour contribuer au projet :

1. Testez le système avec `python test_simple_grasp.py`
2. Lancez un entraînement court pour vérifier le fonctionnement
3. Proposez des améliorations via des issues ou pull requests

---

**Bon entraînement ! 🎯🤖**