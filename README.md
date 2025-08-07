# 🤖 Système de Grasping Robuste avec Agent SAC

> **Agent d'apprentissage par renforcement ultra-professionnel pour grasping avec physique réaliste**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3+-green.svg)](https://mujoco.org)
[![SAC](https://img.shields.io/badge/Algorithm-SAC-orange.svg)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

## 🎯 Aperçu

Ce projet implémente un **système de grasping intelligent** utilisant l'algorithme **Soft Actor-Critic (SAC)** pour apprendre à un robot humanoïde G1 à saisir et manipuler des objets de manière robuste.

### 🌟 Fonctionnalités Principales

- **🤝 Physics Collision Réaliste** - Les bras ne peuvent pas traverser les objets
- **👋 Détection de Contact Précise** - Capteurs sur doigts et palm
- **🔒 Fixation Optimale** - Alignement intelligent palm-cube
- **✊ Contrôle de Force Adaptatif** - Fermeture progressive des doigts
- **🎬 Enregistrement Vidéo Automatique** - Documentation visuelle complète
- **📊 Monitoring Temps Réel** - Métriques et courbes d'apprentissage
- **🧠 Curriculum Learning** - Progression automatique de difficulté
- **💾 Sauvegarde Intelligente** - Modèles et métriques auto-sauvegardés

## 🏗️ Architecture du Système

```
🤖 Robot G1 Humanoïde
├── 💪 Bras Gauche (7 DOF)
├── 💪 Bras Droit (7 DOF)  
├── 👋 Main Gauche (4 doigts)
├── 👋 Main Droite (4 doigts)
└── 📡 Capteurs de Contact (10)

🎯 Environnement MuJoCo
├── 🏓 Table Physique
├── 📦 Cube Manipulable
├── 🔄 Physics Engine Ultra-Stable
└── 📹 Système de Rendu

🧠 Agent SAC
├── 🎛️ Espace Action: 22D (bras + doigts)
├── 👁️ Espace Observation: 88D 
├── 🎓 Curriculum Learning: 7 phases
└── 🎬 Enregistrement Vidéo Intégré
```

## 🚀 Installation Rapide

### Prérequis
- Python 3.8+
- Système Linux/macOS/Windows
- 4GB RAM minimum
- GPU optionnel (CPU supporté)

### Installation Automatique

```bash
# Cloner le repository
git clone https://github.com/votre-nom/robust-grasping.git
cd robust-grasping

# Installation automatique des dépendances
python3 train_robust_grasp.py --demo-only
```

Le script installe automatiquement:
- `numpy` - Calculs numériques
- `gymnasium` - Interface RL
- `mujoco` - Moteur physique
- `stable-baselines3` - Algorithmes RL
- `matplotlib` - Visualisations
- `imageio` - Enregistrement vidéo
- `opencv-python` - Traitement d'images

## 🎮 Utilisation

### 🎭 Test Rapide (Démonstration)

```bash
python3 train_robust_grasp.py --demo-only
```
Entraînement de démonstration de 1000 timesteps (~2 minutes)

### 🏋️ Entraînement Complet

```bash
python3 train_robust_grasp.py --timesteps 500000
```
Entraînement professionnel de 500K timesteps (~2-4 heures)

### ⚙️ Configuration Personnalisée

```bash
python3 train_robust_grasp.py \
  --timesteps 1000000 \
  --lr 1e-4 \
  --buffer 200000 \
  --batch 512 \
  --results-dir /mon/dossier/resultats
```

### 📋 Options Disponibles

| Option | Défaut | Description |
|--------|--------|-------------|
| `--timesteps` | 500,000 | Nombre total de pas d'entraînement |
| `--lr` | 3e-4 | Taux d'apprentissage |
| `--buffer` | 100,000 | Taille du buffer de replay |
| `--batch` | 256 | Taille du batch |
| `--gamma` | 0.99 | Facteur de discount |
| `--tau` | 0.005 | Taux de mise à jour target network |
| `--results-dir` | `/workspace/sac_grasp_results` | Dossier de sauvegarde |
| `--demo-only` | False | Mode démonstration rapide |

## 📊 Phases d'Apprentissage

Le système utilise un **curriculum learning** avec 7 phases progressives:

### 1. 🔍 SEARCH - Recherche du Cube
- **Objectif**: Explorer l'espace et localiser le cube
- **Durée**: 100 timesteps
- **Récompense**: Distance inverse au cube

### 2. 🎯 APPROACH - Approche Contrôlée  
- **Objectif**: S'approcher du cube avec précision
- **Durée**: 80 timesteps
- **Récompense**: Proximité optimale

### 3. 🤝 CONTACT - Contact Initial
- **Objectif**: Établir le premier contact
- **Durée**: 60 timesteps  
- **Récompense**: Détection de contact

### 4. 🔗 ALIGN - Alignement Palm-Cube
- **Objectif**: Positionner optimalement la palm
- **Durée**: 40 timesteps
- **Récompense**: Alignement géométrique

### 5. ✊ GRASP - Saisie Contrôlée
- **Objectif**: Fermer les doigts avec contrôle de force
- **Durée**: 60 timesteps
- **Récompense**: Force de saisie progressive

### 6. ⬆️ LIFT - Levée du Cube
- **Objectif**: Soulever le cube de la table
- **Durée**: 40 timesteps
- **Récompense**: Hauteur du cube

### 7. 💪 HOLD - Maintien Stable
- **Objectif**: Maintenir le cube en l'air
- **Durée**: 60 timesteps
- **Récompense**: Stabilité temporelle

## 📁 Structure des Résultats

Après l'entraînement, le système génère automatiquement:

```
📂 sac_grasp_results/
├── 🧠 models/
│   ├── best_model.zip        # Meilleur modèle SAC
│   ├── final_model.zip       # Modèle final
│   └── eval_*.zip           # Modèles d'évaluation
├── 🎬 videos/
│   ├── demo_episode_01.mp4   # Vidéos de démonstration
│   ├── demo_episode_02.mp4
│   ├── demo_episode_03.mp4
│   └── grasp_episode_*.mp4   # Vidéos d'entraînement
├── 📊 plots/
│   └── learning_curves.png   # Courbes d'apprentissage
├── 📈 metrics/
│   └── training_metrics.json # Métriques détaillées
├── 📝 logs/
│   ├── monitor.csv          # Logs d'entraînement
│   ├── eval_monitor.csv     # Logs d'évaluation
│   └── SAC_Grasping_*/      # Logs TensorBoard
└── 📋 final_report.md        # Rapport complet
```

## 🎬 Test du Modèle Entraîné

```python
from stable_baselines3 import SAC
from robust_grasp_env import RobustGraspEnv

# Charger le meilleur modèle
model = SAC.load('/workspace/sac_grasp_results/models/best_model.zip')

# Créer l'environnement
env = RobustGraspEnv(render_mode='rgb_array', record_video=True)

# Tester le modèle
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Phase: {info['phase']}, Reward: {reward:.2f}")
    
    if done or truncated:
        print(f"Épisode terminé! Cube saisi: {info['cube_grasped']}")
        break

# Sauvegarder la vidéo
env.save_video('test_grasp.mp4')
env.close()
```

## 📊 Monitoring et Visualisation

### TensorBoard
```bash
tensorboard --logdir /workspace/sac_grasp_results/logs
```

### Métriques Temps Réel
Le système affiche automatiquement:
- **Récompense par épisode**
- **Taux de succès**
- **Phase courante**
- **Contact détecté**
- **Force de saisie**

### Courbes d'Apprentissage
Génération automatique de:
- **Récompenses par épisode**
- **Distribution des récompenses**  
- **Taux de succès temporel**
- **Statistiques récapitulatives**

## 🔧 Composants Techniques

### Environnement (`robust_grasp_env.py`)
- **Classe**: `RobustGraspEnv`
- **Base**: `gymnasium.Env`
- **Physics**: MuJoCo 3.3+
- **Observations**: 88 dimensions
- **Actions**: 22 dimensions (continues)

### Agent SAC (`sac_grasp_trainer.py`)
- **Algorithme**: Soft Actor-Critic
- **Bibliothèque**: Stable-Baselines3
- **Hyperparamètres**: Optimisés pour grasping
- **Callbacks**: Monitoring et sauvegarde automatique

### Script Principal (`train_robust_grasp.py`)
- **Interface**: Ligne de commande intuitive
- **Dependencies**: Installation automatique
- **Monitoring**: Temps réel
- **Outputs**: Professionnels et complets

## 🎯 Spécifications Techniques

### Espace d'Action (22D)
```python
action_space = Box(-1.0, 1.0, shape=(22,))
# Bras gauche: 7 DOF (épaule, coude, poignet)
# Bras droit: 7 DOF (épaule, coude, poignet)  
# Doigts gauche: 4 DOF (pouce, index, majeur, annulaire)
# Doigts droit: 4 DOF (pouce, index, majeur, annulaire)
```

### Espace d'Observation (88D)
```python
observation_space = Box(-inf, inf, shape=(88,))
# Positions joints: 36D (qpos)
# Vitesses joints: 36D (qvel)
# Position cube: 3D (x, y, z)
# Orientation cube: 4D (quaternion)
# Vitesse cube: 3D (vx, vy, vz)
# Phase courante: 1D
# État contacts: 5D (palm + doigts + métriques)
```

### Physique MuJoCo
- **Timestep**: 0.002s (500 Hz)
- **Intégrateur**: RK4
- **Solver**: Iterations=50, Tolerance=1e-10
- **Collisions**: Cone elliptique
- **Friction**: Adaptative selon les matériaux

## 🏆 Performances Attendues

### Métriques de Succès
- **Taux de Contact**: >80% des épisodes
- **Taux de Saisie**: >60% des épisodes  
- **Taux de Levée**: >40% des épisodes
- **Stabilité**: >20% des épisodes

### Récompenses Typiques
- **Phase SEARCH**: 0-5 points
- **Phase APPROACH**: 5-10 points
- **Phase CONTACT**: 10-20 points
- **Phase ALIGN**: 20-30 points
- **Phase GRASP**: 30-50 points
- **Phase LIFT**: 50-80 points
- **Phase HOLD**: 80-150 points

### Temps de Convergence
- **Apprentissage initial**: 50K timesteps
- **Performance stable**: 200K timesteps
- **Performance optimale**: 500K timesteps

## 🐛 Dépannage

### Problèmes Courants

#### 1. Erreur de dépendances
```bash
❌ ModuleNotFoundError: No module named 'mujoco'
```
**Solution**: Le script installe automatiquement les dépendances
```bash
python3 train_robust_grasp.py --demo-only
```

#### 2. Erreur de mémoire
```bash
❌ CUDA out of memory
```
**Solution**: Réduire la taille du batch
```bash
python3 train_robust_grasp.py --batch 128
```

#### 3. Entraînement lent
```bash
⚠️ Très lent sur CPU
```
**Solution**: Utiliser un GPU ou réduire les timesteps
```bash
python3 train_robust_grasp.py --timesteps 100000
```

#### 4. Dimension d'observation incorrecte
```bash
❌ Unexpected observation shape (87,) for Box environment
```
**Solution**: Ce problème est résolu dans cette version robuste ✅

### Logs de Debug
En cas de problème, consultez:
- `/workspace/sac_grasp_results/crash_report.json`
- `/workspace/sac_grasp_results/logs/monitor.csv`
- Console output complet

## 🤝 Contribution

Ce projet est conçu pour être **robuste et autonome**. Les améliorations possibles:

1. **Nouveaux Objets**: Ajouter différentes formes à saisir
2. **Multi-Robot**: Support de plusieurs robots
3. **Environnements**: Nouvelles scènes de manipulation
4. **Algorithmes**: Test d'autres méthodes RL (PPO, TD3)
5. **Métriques**: Nouvelles mesures de performance

## 📜 Licence

MIT License - Voir [LICENSE](LICENSE) pour détails

## 📞 Support

Pour toute question ou problème:
1. Consultez ce README en détail
2. Vérifiez les logs générés automatiquement  
3. Lancez un test rapide avec `--demo-only`
4. Créez une issue avec logs complets

## 🎊 Conclusion

Ce système de grasping représente un **état de l'art** en apprentissage par renforcement pour la manipulation robotique:

- ✅ **Robuste**: Gestion d'erreurs complète
- ✅ **Professionnel**: Code clean et documenté  
- ✅ **Autonome**: Installation et exécution automatiques
- ✅ **Complet**: Entraînement, test, et visualisation
- ✅ **Éducatif**: Documentation détaillée et exemples

**🤖 Votre robot apprendra à saisir des objets comme un professionnel !**

---

*Développé avec ❤️ pour la communauté robotique et RL*