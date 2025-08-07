# 🤖 Système de Grasping Robotique avec Curriculum Learning et SAC

## 🎯 Description

Ce projet implémente un système d'apprentissage par renforcement pour l'entraînement d'un robot humanoïde G1 à effectuer des tâches de saisie (grasping) d'objets. Le système utilise l'algorithme **SAC (Soft Actor-Critic)** avec une approche de **Curriculum Learning** pour progresser graduellement de tâches simples vers des tâches complexes.

## 🌟 Caractéristiques Principales

### 🎓 Curriculum Learning Adaptatif
- **Niveau 1**: Stabilisation des bras uniquement
- **Niveau 2**: Stabilisation + Approche du cube
- **Niveau 3**: Stabilisation + Approche + Contact avec l'objet
- **Niveau 4**: Grasping complet (toutes les phases)
- **Niveau 5**: Grasping avec perturbations aléatoires

### 🧠 Agent SAC Optimisé
- Algorithme Soft Actor-Critic avec exploration optimale
- Réseaux de neurones adaptatifs selon le niveau de curriculum
- Buffer de replay intelligent avec priorisation

### 🎬 Capture Vidéo Automatique
- Génération automatique de vidéos de démonstration
- Export en format MP4 et GIF
- Visualisation des performances en temps réel

### 🔧 Physique Ultra-Stable
- Simulation MuJoCo avec paramètres optimisés
- Détection de contact robuste
- Collision physique réaliste entre robot, cube et table

## 📁 Structure du Projet

```
workspace/
├── envs/                           # Environnements de simulation
│   ├── curriculum_grasp_env.py    # Environnement principal avec curriculum
│   └── ...                        # Autres environnements
├── assets/                        # Ressources du modèle
│   ├── hands/                     # Modèles XML des mains
│   │   ├── g1_body.xml           # Corps du robot
│   │   └── g1_fingers.xml        # Doigts et articulations
│   └── ...
├── results/                       # Modèles et résultats
│   └── g1_combined.xml           # Modèle complet du robot
├── train_curriculum_sac_grasp.py # Script d'entraînement principal
├── requirements.txt               # Dépendances Python
└── README.md                     # Cette documentation
```

## 🚀 Installation et Configuration

### Prérequis
- Ubuntu 20.04+ ou système Linux compatible
- Python 3.8+
- GPU CUDA (optionnel mais recommandé)

### Installation

1. **Cloner le projet**
```bash
git clone <repo-url>
cd workspace
```

2. **Créer l'environnement virtuel**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**
```bash
python3 -c "from envs.curriculum_grasp_env import CurriculumGraspEnv; print('✅ Installation réussie')"
```

## 🏃‍♂️ Utilisation

### Entraînement Complet
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'entraînement avec curriculum learning
python3 train_curriculum_sac_grasp.py
```

### Paramètres d'Entraînement
Le script d'entraînement accepte plusieurs paramètres configurables :

- **total_timesteps**: Nombre total de pas d'entraînement (défaut: 100,000)
- **learning_rate**: Taux d'apprentissage SAC (défaut: 0.0005)
- **buffer_size**: Taille du buffer de replay (défaut: 50,000)

## 📊 Résultats et Métriques

### Dossier de Sortie
Tous les résultats sont sauvegardés dans `/workspace/curriculum_sac_results/`:

```
curriculum_sac_results/
├── models/                    # Modèles SAC sauvegardés
│   ├── curriculum_sac_model_level_1.zip
│   ├── curriculum_sac_model_level_2.zip
│   └── ...
├── logs/                      # Logs d'entraînement TensorBoard
├── videos/                    # Vidéos de démonstration
│   ├── demonstration.mp4      # Vidéo principale
│   └── demonstration.gif      # Version GIF
├── training_metrics.json     # Métriques détaillées
└── curriculum_summary.txt    # Résumé lisible
```

### Métriques Suivies
- **Récompenses par niveau** de curriculum
- **Taux de succès** pour chaque phase
- **Temps de convergence** par niveau
- **Transitions de niveau** automatiques
- **Stabilité de l'entraînement**

## 🎮 Phases de Grasping

### 1. STABILIZE (Stabilisation)
- Apprentissage du contrôle moteur de base
- Maintien des bras en position stable
- Réduction des oscillations

### 2. APPROACH (Approche)
- Navigation vers l'objet cible
- Coordination bimanuelle
- Planification de trajectoire

### 3. CONTACT (Contact)
- Détection tactile avec l'objet
- Positionnement précis des doigts
- Contact physique robuste

### 4. GRASP (Saisie)
- Fermeture coordonnée des doigts
- Application de force optimale
- Stabilisation de la prise

### 5. LIFT (Soulèvement)
- Soulèvement de l'objet
- Maintien de la prise
- Contrôle de l'équilibre

### 6. HOLD (Maintien)
- Maintien prolongé de l'objet
- Résistance aux perturbations
- Stabilité à long terme

## 🔧 Configuration Avancée

### Modification des Paramètres de Curriculum

Dans `envs/curriculum_grasp_env.py`, vous pouvez ajuster :

```python
# Seuils de réussite pour progression
'success_threshold': 15.0,  # Récompense requise
'episodes_required': 5,     # Épisodes consécutifs réussis

# Durées des phases
'max_episode_steps': 1000,  # Durée maximale d'épisode
```

### Paramètres Physiques

Dans le modèle XML (`results/g1_combined.xml`) :

```xml
<!-- Paramètres de simulation -->
<option timestep="0.0005" iterations="500" solver="Newton" tolerance="1e-12">

<!-- Propriétés du cube -->
<geom name="cube_geom" type="box" size="0.025 0.025 0.025" 
      friction="2.5 0.4 0.2" mass="0.05"/>
```

## 📈 Monitoring et Visualisation

### TensorBoard
```bash
# Visualiser les métriques d'entraînement
tensorboard --logdir=/workspace/curriculum_sac_results/logs
```

### Métriques Clés
- `train/reward`: Récompense moyenne par épisode
- `train/success_rate`: Taux de succès du grasping
- `curriculum/level`: Niveau actuel du curriculum
- `curriculum/transitions`: Transitions entre niveaux

## 🚨 Dépannage

### Problèmes Courants

1. **Erreur de dimension d'observation**
   - Solution : L'environnement s'ajuste automatiquement

2. **Problèmes de rendu**
   - Vérifier l'installation d'OpenGL : `sudo apt install libgl1-mesa-glx`

3. **Performance lente**
   - Utiliser un GPU CUDA si disponible
   - Réduire `total_timesteps` pour des tests rapides

### Logs de Debug
Les logs détaillés sont disponibles dans :
- Console : Messages en temps réel
- Fichiers : `/workspace/curriculum_sac_results/logs/`

## 🤝 Contribution

### Structure de Code
- **Environnements** : `envs/`
- **Scripts d'entraînement** : Racine du projet
- **Assets** : `assets/`
- **Tests** : `tests/`

### Standards de Code
- Documentation complète des fonctions
- Messages de log informatifs avec emojis
- Gestion robuste des erreurs
- Code modulaire et réutilisable

## 📄 Licence

Ce projet est développé à des fins de recherche et d'éducation en robotique et apprentissage par renforcement.

## 🎯 Objectifs Futurs

- [ ] Support multi-objets
- [ ] Grasping bimanuel coordonné  
- [ ] Apprentissage par imitation
- [ ] Déploiement sur robot réel
- [ ] Interface utilisateur graphique

---

**Développé avec ❤️ pour l'avancement de la robotique intelligente**