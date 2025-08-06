# Système d'Entraînement de Saisie G1

## 🎯 Objectif

Ce système d'entraînement permet au robot G1 d'apprendre à saisir un cube en utilisant l'apprentissage par renforcement (SAC) avec détection de contact par capteurs de force.

## 📁 Structure du Projet

```
/workspace/
├── envs/
│   └── simple_grasp_env.py          # Environnement de saisie simplifié
├── agents/
│   └── improved_sac_agent.py        # Agent SAC amélioré
├── utils/
│   └── video_recorder.py            # Enregistrement vidéo
├── train_simple_grasp.py            # Script d'entraînement principal
├── test_simple_grasp_basic.py       # Tests de base
└── results/
    └── g1_combined.xml              # Modèle MuJoCo du robot G1
```

## 🚀 Installation

### 1. Dépendances Système
```bash
# Installer les dépendances MuJoCo (si nécessaire)
sudo apt update
sudo apt install python3-pip python3-venv
```

### 2. Dépendances Python
```bash
# Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate

# Ou installer directement (si pas de venv disponible)
pip install --break-system-packages numpy torch gymnasium mujoco matplotlib imageio
```

### 3. Vérification
```bash
python3 test_simple_grasp_basic.py
```

## 🎮 Utilisation

### Lancement de l'Entraînement

```bash
# Entraînement standard (2000 épisodes)
python3 train_simple_grasp.py

# Entraînement court pour test
python3 train_simple_grasp.py --episodes 100

# Personnaliser les paramètres
python3 train_simple_grasp.py \
    --episodes 1000 \
    --lr 1e-4 \
    --output /workspace/my_training \
    --curriculum 1
```

### Paramètres Disponibles

- `--episodes`: Nombre d'épisodes d'entraînement (défaut: 2000)
- `--lr`: Taux d'apprentissage (défaut: 3e-4)
- `--output`: Dossier de sortie (défaut: /workspace/training_results)
- `--curriculum`: Niveau de curriculum initial (défaut: 1)
- `--video`: Activer l'enregistrement vidéo

## 🏗️ Architecture du Système

### 1. Environnement de Saisie (`SimpleGraspEnv`)

**Caractéristiques:**
- Détection de contact via capteurs de force (pas de tip sensors)
- Phases automatiques: approche → contact → saisie → levage
- Curriculum learning intégré (3 niveaux de difficulté)
- Observation complète: positions joints, vitesses, position cube, capteurs force

**Espace d'Observation:**
- Positions des joints (nq dimensions)
- Vitesses des joints (nv dimensions) 
- Position du cube (3 dimensions)
- Hauteur relative du cube (1 dimension)
- Données capteurs de force (n_sensors dimensions)
- Phase actuelle (4 dimensions, one-hot)

**Espace d'Action:**
- Contrôle direct des actuateurs [-1, 1]

### 2. Système de Récompenses

```python
# Récompense de contact
if contact_detected:
    reward += min(1.0, total_force * 0.5)
else:
    reward -= 0.1

# Récompense de hauteur (levage)
height_diff = cube_height - initial_height
if height_diff > 0.01:
    reward += min(10.0, height_diff * 20)

# Pénalités
reward -= 0.01 * action_energy  # Mouvement excessif
if cube_falls:
    reward -= 5.0
```

### 3. Agent SAC Amélioré

**Caractéristiques:**
- Réseaux Actor/Critic avec target networks
- Automatic temperature tuning (alpha)
- Replay buffer de 100k transitions
- Architecture: [256, 256] par défaut
- Optimisation Adam avec lr=3e-4

**Composants:**
- `Actor`: Réseau de politique stochastique avec reparameterization trick
- `Critic`: Double Q-networks pour réduire l'overestimation
- `ReplayBuffer`: Stockage efficace des transitions

### 4. Curriculum Learning

**Niveaux:**
1. **Niveau 1**: Récompenses standard
2. **Niveau 2**: Récompenses × 1.2 (plus difficile)
3. **Niveau 3**: Récompenses × 1.5 (le plus difficile)

**Progression:**
- Seuil de succès: 70% sur 200 épisodes
- Passage automatique au niveau suivant
- Redescente possible si échec prolongé

## 📊 Sorties et Métriques

### Structure des Résultats
```
training_results/
├── models/
│   ├── checkpoint_episode_200.pth
│   ├── checkpoint_episode_400.pth
│   └── final_model.pth
├── videos/
│   ├── episode_100_*.mp4
│   ├── training_compilation_*.mp4
│   └── *.txt (métadonnées)
├── logs/
│   ├── metrics_episode_*.json
│   ├── final_metrics.json
│   └── training_plots.png
```

### Métriques Suivies
- Récompenses par épisode
- Longueurs des épisodes  
- Taux de succès
- Taux de contact
- Pertes d'entraînement (actor/critic)
- Paramètre alpha (température)

### Enregistrements Vidéo
- Vidéo d'évaluation tous les 100 épisodes
- Compilation finale de 5 épisodes
- Métadonnées détaillées (récompenses, succès, contact)
- Format MP4, 30 FPS

## 🔧 Configuration Avancée

### Modifier les Paramètres

Éditer le fichier `train_simple_grasp.py`, fonction `load_config()`:

```python
config = {
    # Environnement
    'model_path': '/workspace/results/g1_combined.xml',
    'max_episode_steps': 500,
    'curriculum_level': 1,
    
    # Entraînement
    'total_episodes': 2000,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 100000,
    'hidden_sizes': [256, 256],
    
    # Curriculum
    'curriculum_threshold': 0.7,
    'episodes_per_level': 200,
    
    # Logging
    'log_interval': 50,
    'save_interval': 200,
    'video_interval': 100,
}
```

### Personnaliser les Récompenses

Modifier la méthode `_compute_reward()` dans `SimpleGraspEnv`:

```python
def _compute_reward(self):
    reward = 0.0
    
    # Vos récompenses personnalisées
    contact_reward = self._compute_contact_reward()
    height_reward = self._compute_height_reward()
    stability_reward = self._compute_stability_reward()
    
    return contact_reward + height_reward + stability_reward
```

## 🐛 Résolution de Problèmes

### Erreurs Communes

1. **Modèle non trouvé**
   ```bash
   ❌ Modèle non trouvé: /workspace/results/g1_combined.xml
   💡 Créez d'abord le modèle avec: python create_combined_model.py
   ```

2. **Dépendances manquantes**
   ```bash
   ModuleNotFoundError: No module named 'numpy'
   # Solution: pip install numpy torch gymnasium mujoco
   ```

3. **Environnement externally managed**
   ```bash
   # Solution: utiliser --break-system-packages ou créer un venv
   pip install --break-system-packages [packages]
   ```

4. **Erreurs MuJoCo**
   - Vérifier que le modèle XML est valide
   - S'assurer que les capteurs sont bien définis
   - Vérifier les chemins de fichiers dans le XML

### Optimisation des Performances

1. **GPU**: Le système détecte automatiquement CUDA
2. **Mémoire**: Réduire `buffer_size` si RAM limitée
3. **Vitesse**: Réduire `hidden_sizes` pour des réseaux plus petits
4. **Vidéos**: Désactiver pour accélérer l'entraînement

## 📈 Interprétation des Résultats

### Signaux de Succès
- Récompenses croissantes au fil des épisodes
- Taux de contact élevé (>80%)
- Cube soulevé régulièrement
- Pertes d'entraînement qui se stabilisent

### Signaux d'Échec
- Récompenses qui stagnent ou décroissent
- Aucun contact avec le cube
- Actions erratiques dans les vidéos
- Explosion des pertes

### Ajustements Recommandés
- **Apprentissage lent**: Augmenter le learning rate
- **Instabilité**: Réduire le learning rate
- **Pas de contact**: Ajuster les récompenses de contact
- **Pas de progrès**: Vérifier le modèle et les capteurs

## 🎯 Objectifs d'Amélioration

### Court Terme
- [ ] Optimiser les hyperparamètres
- [ ] Ajouter plus de variabilité dans les positions initiales
- [ ] Implémenter des récompenses plus sophistiquées

### Long Terme  
- [ ] Support multi-objets
- [ ] Saisie bi-manuelle coordonnée
- [ ] Transfert vers robot réel
- [ ] Interface de monitoring temps réel

## 📞 Support

Pour des questions ou problèmes:
1. Vérifier ce README
2. Lancer `python3 test_simple_grasp_basic.py`
3. Vérifier les logs dans `training_results/logs/`
4. Examiner les vidéos pour diagnostiquer les comportements

---

**Version**: 1.0  
**Auteur**: Assistant IA  
**Projet**: G1 Fingers Manipulation