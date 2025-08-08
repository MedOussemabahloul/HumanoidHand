# 🎯 Système d'Entraînement Robuste pour Grasping G1

## 📋 Vue d'ensemble

Ce système d'entraînement robuste corrige tous les problèmes identifiés dans les versions précédentes :

- ✅ **Vitesses excessives** - Contrôle de vitesse intelligent
- ✅ **Erreurs mujoco** - Gestion robuste des imports et contextes
- ✅ **Capture vidéo** - Système de vidéo intégré et fonctionnel
- ✅ **Stagnation** - Système de récompenses adaptatif
- ✅ **Instabilité** - Physique ultra-stable
- ✅ **Monitoring** - Suivi en temps réel des performances

## 🏗️ Architecture

### Fichiers principaux

1. **`envs/robust_curriculum_grasp_env.py`** - Environnement robuste avec curriculum learning
2. **`train_robust_curriculum_sac.py`** - Script d'entraînement principal
3. **`test_robust_environment.py`** - Tests de l'environnement
4. **`run_robust_training.py`** - Lanceur principal
5. **`README_ROBUST_TRAINING.md`** - Ce fichier

### Fonctionnalités avancées

- **Curriculum Learning Adaptatif** : Progression automatique de difficulté
- **Contrôle de Vitesse Intelligent** : Évite les vitesses excessives
- **Capture Vidéo Intégrée** : Génération automatique de vidéos
- **Simulation Mujoco en Temps Réel** : Ouverture automatique du viewer
- **Monitoring Avancé** : Suivi des performances en temps réel
- **Sauvegarde Intelligente** : Modèles sauvegardés par niveau

## 🚀 Installation et Configuration

### Prérequis

```bash
# Packages Python requis
pip install numpy gymnasium stable-baselines3 mujoco opencv-python matplotlib

# Vérifier que le fichier modèle existe
ls /home/oussema/Documents/project/results/g1_combined.xml
```

### Structure des dossiers

```
/home/oussema/Documents/project/
├── envs/
│   └── robust_curriculum_grasp_env.py
├── results/
│   └── g1_combined.xml
├── robust_curriculum_sac_results/
│   ├── models/
│   ├── videos/
│   ├── logs/
│   └── plots/
├── train_robust_curriculum_sac.py
├── test_robust_environment.py
├── run_robust_training.py
└── README_ROBUST_TRAINING.md
```

## 🎯 Utilisation

### 1. Lancement rapide (recommandé)

```bash
# Lancer le système complet
python3 run_robust_training.py
```

Cette commande :
- Vérifie les dépendances
- Lance les tests de l'environnement
- Démarre l'entraînement avec curriculum learning
- Ouvre automatiquement la simulation Mujoco
- Génère les vidéos de démonstration
- Sauvegarde tous les résultats

### 2. Lancement avec options

```bash
# Passer les tests
python3 run_robust_training.py --skip-tests

# Passer l'entraînement (pour tester seulement)
python3 run_robust_training.py --skip-training

# Ouvrir les résultats après
python3 run_robust_training.py --open-results
```

### 3. Lancement manuel

```bash
# Test de l'environnement
python3 test_robust_environment.py

# Entraînement direct
python3 train_robust_curriculum_sac.py
```

## 📊 Curriculum Learning

Le système utilise un curriculum learning en 5 niveaux :

### Niveau 1 : Stabilisation
- **Objectif** : Apprendre à stabiliser les bras
- **Phases** : STABILIZE uniquement
- **Durée** : 200 steps max
- **Seuil de succès** : 15.0

### Niveau 2 : Approche
- **Objectif** : Apprendre à approcher le cube
- **Phases** : STABILIZE + APPROACH
- **Durée** : 300 steps max
- **Seuil de succès** : 25.0

### Niveau 3 : Contact
- **Objectif** : Apprendre à toucher le cube
- **Phases** : STABILIZE + APPROACH + CONTACT
- **Durée** : 400 steps max
- **Seuil de succès** : 40.0

### Niveau 4 : Grasping Complet
- **Objectif** : Grasping complet
- **Phases** : Toutes les phases
- **Durée** : 500 steps max
- **Seuil de succès** : 60.0

### Niveau 5 : Niveau Maître
- **Objectif** : Grasping avec perturbations
- **Phases** : Toutes les phases + bruit
- **Durée** : 500 steps max
- **Seuil de succès** : 80.0

## 🎥 Capture Vidéo

Le système génère automatiquement des vidéos :

### Vidéos d'entraînement
- **Emplacement** : `/home/oussema/Documents/project/robust_curriculum_sac_results/videos/`
- **Format** : MP4
- **Résolution** : 640x480
- **FPS** : 30

### Vidéo finale
- **Nom** : `final_demo.mp4`
- **Contenu** : Démonstration du modèle final
- **Ouverture automatique** : Oui

## 🔧 Configuration Avancée

### Paramètres d'environnement

```python
# Dans robust_curriculum_grasp_env.py
curriculum_levels = {
    1: {
        'max_velocity': 2.0,      # Vitesse maximale
        'action_scale': 0.1,      # Échelle des actions
        'success_threshold': 15.0, # Seuil de succès
        'episodes_required': 5     # Épisodes requis
    }
    # ... autres niveaux
}
```

### Paramètres d'entraînement

```python
# Dans train_robust_curriculum_sac.py
base_params = {
    'learning_rate': 0.0001,  # Taux d'apprentissage
    'buffer_size': 50000,     # Taille du buffer
    'batch_size': 128,        # Taille du batch
    'gamma': 0.98,            # Facteur de discount
    'ent_coef': 0.2           # Coefficient d'entropie
}
```

## 📈 Monitoring et Résultats

### Métriques suivies

- **Récompenses par épisode** : Progression de l'apprentissage
- **Niveaux de curriculum** : Progression dans les niveaux
- **Taux de succès** : Pourcentage d'épisodes réussis
- **Vitesses moyennes** : Stabilité du robot
- **Temps d'entraînement** : Durée totale

### Fichiers de résultats

```
robust_curriculum_sac_results/
├── models/
│   ├── level_1_final.zip
│   ├── level_2_final.zip
│   └── ...
├── videos/
│   ├── grasp_training_20241201_143022.mp4
│   └── final_demo.mp4
├── logs/
│   ├── training_metrics.json
│   └── tensorboard_logs/
└── plots/
    └── curriculum_progress.png
```

## 🐛 Dépannage

### Problèmes courants

#### 1. Erreur "mujoco referenced before assignment"
**Solution** : Utilisez le nouvel environnement robuste qui gère correctement les imports.

#### 2. Vitesses excessives
**Solution** : Le système de contrôle de vitesse intelligent corrige automatiquement ce problème.

#### 3. Vidéo ne s'ouvre pas
**Solution** : Vérifiez que OpenCV est installé et que le dossier videos/ existe.

#### 4. Stagnation de l'apprentissage
**Solution** : Le système de récompenses adaptatif et le curriculum learning évitent ce problème.

### Logs et débogage

```bash
# Vérifier les logs
tail -f /home/oussema/Documents/project/robust_curriculum_sac_results/logs/training.log

# Vérifier les métriques
cat /home/oussema/Documents/project/robust_curriculum_sac_results/training_metrics.json
```

## 🎯 Performances Attendues

### Métriques de succès

- **Récompense moyenne** : > 50.0 (niveau 4+)
- **Taux de succès** : > 70%
- **Vitesse moyenne** : < 5.0
- **Temps d'entraînement** : 2-4 heures

### Progression typique

```
Niveau 1: 0-50 épisodes (stabilisation)
Niveau 2: 50-150 épisodes (approche)
Niveau 3: 150-300 épisodes (contact)
Niveau 4: 300-500 épisodes (grasping)
Niveau 5: 500+ épisodes (maîtrise)
```

## 🔄 Mise à jour et Maintenance

### Mise à jour du système

```bash
# Sauvegarder les résultats existants
cp -r robust_curriculum_sac_results robust_curriculum_sac_results_backup

# Mettre à jour les scripts
git pull origin main

# Relancer les tests
python3 test_robust_environment.py
```

### Maintenance

- **Nettoyage** : Supprimer les fichiers temporaires
- **Sauvegarde** : Sauvegarder les modèles entraînés
- **Monitoring** : Vérifier les performances régulièrement

## 📞 Support

Pour toute question ou problème :

1. Vérifiez les logs dans `/home/oussema/Documents/project/robust_curriculum_sac_results/logs/`
2. Lancez les tests avec `python3 test_robust_environment.py`
3. Consultez ce README pour les solutions courantes

## 🎉 Conclusion

Ce système d'entraînement robuste garantit :

- **Stabilité** : Physique ultra-stable et contrôle de vitesse intelligent
- **Performance** : Curriculum learning adaptatif et récompenses optimisées
- **Visibilité** : Capture vidéo automatique et monitoring en temps réel
- **Robustesse** : Gestion d'erreurs complète et récupération automatique

**Bonne entraînement ! 🚀**