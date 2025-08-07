# 🚀 Guide d'Utilisation Rapide

## 🎯 Comment Utiliser le Système

### 1. Installation Rapide
```bash
# Cloner le projet et aller dans le dossier
cd /workspace

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Test Rapide du Système
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Test rapide (1000 timesteps)
python3 test_quick_training.py
```

### 3. Entraînement Complet Robuste
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Entraînement robuste (50,000 timesteps)
python3 train_robust_curriculum_sac.py
```

## 📊 Résultats

Après l'entraînement, vous trouverez :

```
curriculum_sac_results_YYYYMMDD_HHMMSS/
├── models/
│   └── robust_curriculum_sac_final.zip    # Modèle entraîné
├── videos/
│   ├── demonstration.mp4                  # Vidéo de démonstration
│   └── demonstration.gif                  # Version GIF
├── logs/                                  # Logs TensorBoard
├── training_metrics.json                 # Métriques détaillées
└── training_summary.txt                  # Résumé lisible
```

## 🎮 Phases d'Apprentissage

Le robot apprend progressivement :

1. **🎯 Niveau 1** : Stabilisation des bras
2. **🎯 Niveau 2** : Stabilisation + Approche du cube  
3. **🎯 Niveau 3** : + Détection de contact
4. **🎯 Niveau 4** : Grasping complet
5. **🎯 Niveau 5** : Grasping avec perturbations

## 📈 Monitoring

Visualiser l'entraînement en temps réel :
```bash
tensorboard --logdir=curriculum_sac_results_*/logs
```

## 🔧 Configuration Rapide

Pour modifier l'entraînement, éditez dans `train_robust_curriculum_sac.py` :

```python
# Nombre de timesteps (plus = meilleur mais plus long)
total_timesteps = 50000  # Modifier cette valeur

# Dans create_model(), modifier :
learning_rate=0.0003,    # Vitesse d'apprentissage
buffer_size=100000,      # Taille du buffer
batch_size=256,          # Taille des batches
```

## 🎬 Capture Vidéo

Le système génère automatiquement :
- ✅ Vidéo MP4 de démonstration
- ✅ GIF animé pour visualisation rapide
- ✅ Métriques de performance

## ⚡ Scripts Disponibles

| Script | Description | Durée |
|--------|-------------|-------|
| `test_quick_training.py` | Test rapide | ~2 minutes |
| `train_robust_curriculum_sac.py` | Entraînement complet | ~30 minutes |
| `train_curriculum_sac_grasp.py` | Version originale | Variable |

## 🚨 Dépannage Rapide

**Problème** : Erreur d'observation shape
**Solution** : ✅ Auto-corrigé par le système

**Problème** : Rendu vidéo échoue  
**Solution** : Le système continue sans vidéo

**Problème** : Performance lente
**Solution** : Réduire `total_timesteps` ou utiliser GPU

## 🎯 Objectif Final

Le robot doit apprendre à :
- 🤖 Stabiliser ses bras
- 🔍 Localiser le cube sur la table
- 🤏 Approcher et saisir le cube
- ⬆️ Soulever et maintenir l'objet
- 📹 Tout cela est enregistré automatiquement !

---

**🎉 Prêt à commencer ? Lancez `python3 train_robust_curriculum_sac.py` !**