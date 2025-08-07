# 🚀 Guide d'Utilisation - Système de Grasping SAC

> **Guide complet pour utiliser le système de grasping intelligent avec agent SAC**

## 🎯 Résumé

Ce système permet d'entraîner un robot humanoïde G1 à saisir et manipuler des objets en utilisant l'apprentissage par renforcement (SAC). Le robot apprend automatiquement:

- 🔍 **Rechercher** le cube dans l'environnement
- 🎯 **S'approcher** avec précision
- 🤝 **Détecter le contact** avec les doigts et la palm
- 🔒 **Fixer la palm** au cube de manière optimale
- ✊ **Fermer les doigts** avec contrôle de force
- ⬆️ **Lever le cube** de la table
- 💪 **Maintenir** l'objet stable

## 🚀 Utilisation Rapide

### 1. Entraînement Rapide (5K timesteps - 2 minutes)
```bash
python3 train_final.py --quick
```

### 2. Entraînement Complet (100K timesteps - 20 minutes)
```bash
python3 train_final.py --timesteps 100000
```

### 3. Test du Modèle Entraîné
```bash
python3 test_trained_model.py
```

## 📋 Options d'Entraînement

### Script Principal: `train_final.py`

```bash
# Test rapide
python3 train_final.py --quick

# Entraînement personnalisé
python3 train_final.py --timesteps 200000 --results-dir /mon/dossier

# Aide complète
python3 train_final.py --help
```

**Options disponibles:**
- `--timesteps N`: Nombre de pas d'entraînement (défaut: 100,000)
- `--quick`: Mode rapide 5K timesteps pour test
- `--results-dir PATH`: Dossier de sauvegarde

### Script de Test: `test_trained_model.py`

```bash
# Test avec modèle par défaut
python3 test_trained_model.py

# Test personnalisé
python3 test_trained_model.py --model /chemin/vers/modele.zip --episodes 5

# Test avec sortie personnalisée
python3 test_trained_model.py --output-dir /mes/videos
```

## 📁 Structure des Résultats

Après l'entraînement, vous obtenez:

```
final_results/
├── models/
│   ├── best_model.zip      # 🏆 Meilleur modèle (performance optimale)
│   └── final_model.zip     # 🎯 Modèle final d'entraînement
├── logs/
│   ├── monitor.csv         # 📊 Métriques d'entraînement
│   └── SAC_*/              # 📈 Logs TensorBoard
├── videos/                 # 🎬 Vidéos de démonstration
└── training_report.json    # 📋 Rapport complet
```

## 🎬 Génération de Vidéos

### Vidéos Automatiques
- **Pendant l'entraînement**: Désactivées pour éviter les problèmes de rendu
- **Après l'entraînement**: 3 vidéos de démonstration générées automatiquement

### Vidéos de Test
```bash
# Générer des vidéos de test
python3 test_trained_model.py --episodes 5

# Les vidéos sont sauvegardées dans test_videos/
```

## 🔧 Utilisation Programmatique

### Charger et Utiliser un Modèle

```python
from stable_baselines3 import SAC
from robust_grasp_env import RobustGraspEnv

# Charger le meilleur modèle
model = SAC.load('/workspace/final_results/models/best_model.zip')

# Créer l'environnement
env = RobustGraspEnv(render_mode='rgb_array', record_video=True)

# Tester le modèle
obs, _ = env.reset()
for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Step {step}: Phase {info['phase']}, Reward {reward:.2f}")
    
    if done or truncated:
        print(f"Épisode terminé! Cube saisi: {info['cube_grasped']}")
        break

# Sauvegarder la vidéo
env.save_video('mon_test.mp4')
env.close()
```

### Informations Disponibles

Chaque `info` retourné par `env.step()` contient:

```python
{
    'phase': 'SEARCH|APPROACH|CONTACT|ALIGN|GRASP|LIFT|HOLD',
    'cube_position': [x, y, z],           # Position du cube
    'cube_grasped': True/False,           # Cube saisi ?
    'cube_lifted': True/False,            # Cube levé ?
    'palm_contact': True/False,           # Contact palm-cube ?
    'finger_contacts': 0-8,               # Nombre de contacts doigts
    'grasp_force': 0.0-1.0,              # Force de saisie
    'stability_score': 0.0-1.0,          # Score de stabilité
    'min_distance': float                 # Distance minimale au cube
}
```

## 📊 Interprétation des Résultats

### Récompenses Typiques
- **0-1000**: Robot en apprentissage initial
- **1000-3000**: Performances correctes
- **3000-5000**: Bonnes performances
- **5000+**: Performances excellentes

### Phases de Progression
1. **SEARCH** (0-5 pts): Exploration et localisation
2. **APPROACH** (5-10 pts): Approche contrôlée
3. **CONTACT** (10-20 pts): Premier contact
4. **ALIGN** (20-30 pts): Alignement optimal
5. **GRASP** (30-50 pts): Saisie progressive
6. **LIFT** (50-80 pts): Levée du cube
7. **HOLD** (80-150 pts): Maintien stable

### Taux de Succès
- **80%+**: Robot expert
- **60-80%**: Très bonnes performances
- **40-60%**: Bonnes performances
- **20-40%**: Performances moyennes
- **<20%**: Besoin de plus d'entraînement

## 🐛 Résolution de Problèmes

### Problème: Modèle non trouvé
```bash
❌ Modèle introuvable: /workspace/final_results/models/best_model.zip
```
**Solution**: Lancez d'abord l'entraînement
```bash
python3 train_final.py --quick
```

### Problème: Erreur de mémoire
```bash
❌ CUDA out of memory
```
**Solution**: Utilisez CPU (par défaut) ou réduisez le batch size

### Problème: Entraînement lent
**Solution**: Utilisez le mode rapide pour tester
```bash
python3 train_final.py --quick
```

### Problème: Pas de vidéo générée
**Solution**: Les vidéos sont créées seulement pendant les tests
```bash
python3 test_trained_model.py
```

## ⚡ Conseils d'Optimisation

### Pour un Entraînement Rapide
```bash
# Test de 2 minutes
python3 train_final.py --quick
```

### Pour de Meilleures Performances
```bash
# Entraînement long
python3 train_final.py --timesteps 500000
```

### Pour Analyser l'Apprentissage
```bash
# Monitoring TensorBoard
tensorboard --logdir final_results/logs
```

## 🎯 Exemples d'Utilisation

### Cas 1: Test Rapide du Système
```bash
# 1. Entraînement rapide (2 min)
python3 train_final.py --quick

# 2. Test du modèle (1 min)
python3 test_trained_model.py

# 3. Vérifier les résultats
ls final_results/
ls test_videos/
```

### Cas 2: Entraînement Professionnel
```bash
# 1. Entraînement complet (20-60 min)
python3 train_final.py --timesteps 200000

# 2. Test approfondi
python3 test_trained_model.py --episodes 10

# 3. Analyse des performances
cat final_results/training_report.json
```

### Cas 3: Développement et Debug
```bash
# 1. Test très rapide
python3 train_final.py --quick --results-dir debug_results

# 2. Vérification fonctionnelle
python3 test_trained_model.py --model debug_results/models/best_model.zip --episodes 1
```

## 🎊 Félicitations !

Vous savez maintenant utiliser le système de grasping SAC. Le robot peut apprendre à:

- ✅ Rechercher des objets avec des mouvements naturels
- ✅ Détecter les collisions physiques (ne traverse pas les objets)
- ✅ Contrôler la force de saisie avec les doigts et la palm
- ✅ Maintenir des objets stables en l'air
- ✅ Générer automatiquement des vidéos de démonstration

**🤖 Votre robot est maintenant prêt pour le grasping intelligent !**