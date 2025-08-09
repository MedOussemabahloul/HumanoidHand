# 🎯 Entraînement de Grasping Robotique Simplifié

## 🚀 Version simplifiée et robuste

Ce projet contient une version simplifiée et fonctionnelle de l'entraînement de grasping robotique, inspirée du code de votre collègue mais optimisée pour éviter :

- ❌ La stagnation des rewards
- ❌ Les erreurs NaN/Inf 
- ❌ Les vitesses excessives
- ❌ La complexité inutile du curriculum learning

## 📁 Structure des fichiers

### Fichiers principaux
- `envs/simple_robust_grasp_env.py` - Environnement de grasping simplifié
- `simple_training_td3.py` - Script d'entraînement TD3 robuste
- `evaluate_and_download.py` - Évaluation et génération de vidéos
- `start_training.py` - Script de démarrage rapide

### Fichiers de configuration
- `requirements.txt` - Dépendances Python
- `README_SIMPLE.md` - Ce fichier

## 🛠️ Installation

```bash
# Installer les dépendances
pip install --break-system-packages numpy mujoco gymnasium stable-baselines3[extra] imageio Pillow

# Ou si vous préférez un environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install numpy mujoco gymnasium stable-baselines3[extra] imageio Pillow
```

## 🏃 Utilisation rapide

### 1. Test rapide (recommandé pour commencer)
```bash
python3 start_training.py
```

### 2. Entraînement personnalisé
```bash
python3 simple_training_td3.py
```

### 3. Évaluation et vidéo
```bash
# Mode automatique (comme votre collègue)
python3 evaluate_and_download.py --quick

# Mode complet avec options
python3 evaluate_and_download.py --model simple_td3_results/final_model.zip --video ma_video.mp4
```

## 🎯 Fonctionnalités

### Environnement (`SimpleRobustGraspEnv`)
- ✅ Modèle MuJoCo intégré (pas besoin de fichiers XML externes)
- ✅ Système de récompenses équilibré (inspiré du collègue)
- ✅ Gestion robuste des NaN/Inf
- ✅ Assistance au grasping automatique
- ✅ Mode headless (fonctionne sans affichage)

### Entraînement TD3
- ✅ Configuration testée et stable
- ✅ Sauvegarde automatique tous les 25k steps
- ✅ Monitoring des performances
- ✅ Vidéos d'évaluation automatiques
- ✅ Gestion des interruptions

### Évaluation
- ✅ Reproduction exacte du code de votre collègue
- ✅ Génération de vidéos longues (1000 steps)
- ✅ Statistiques détaillées
- ✅ Export automatique

## 📊 Configuration par défaut

```python
# Entraînement
total_timesteps = 100_000  # Commencer petit pour tester
learning_rate = 3e-4
batch_size = 256
buffer_size = 1_000_000

# Comme votre collègue
tau = 0.02
gamma = 0.98
```

## 🎥 Génération de vidéos

Le système génère automatiquement des vidéos :

1. **Pendant l'entraînement** : Toutes les 25k steps
2. **À la fin** : Vidéo d'évaluation complète
3. **Sur demande** : Avec `evaluate_and_download.py`

## 🔧 Personnalisation

### Modifier la durée d'entraînement
Éditez `simple_training_td3.py`, ligne ~138 :
```python
total_timesteps = 500_000  # Augmenter pour plus d'entraînement
```

### Changer la fréquence de sauvegarde
Éditez `simple_training_td3.py`, ligne ~202 :
```python
save_freq=50000,  # Sauvegarder plus souvent
```

### Modifier le modèle robotique
L'environnement génère automatiquement un modèle simplifié. Pour utiliser votre propre modèle :
```python
env = SimpleRobustGraspEnv(model_path="votre_modele.xml")
```

## 📈 Monitoring

### Logs d'entraînement
```
simple_td3_results/
├── logs/           # TensorBoard
├── videos/         # Vidéos d'évaluation
├── model_25000_steps.zip
├── final_model.zip
└── stats_*.json    # Statistiques
```

### Visualisation TensorBoard
```bash
tensorboard --logdir simple_td3_results/logs
```

## 🐛 Dépannage

### Problème : "gladLoadGL error"
```
Solution : L'environnement fonctionne en mode headless automatiquement.
L'entraînement continue sans rendu visuel.
```

### Problème : Rewards stagnants
```
- L'environnement simplifié évite ce problème
- Système de récompenses testé et équilibré
- Assistance au grasping automatique
```

### Problème : Mémoire insuffisante
```python
# Réduire la taille du buffer
buffer_size = 100_000  # Au lieu de 1_000_000
batch_size = 128       # Au lieu de 256
```

## 🏆 Résultats attendus

Avec la configuration par défaut, vous devriez observer :

1. **Convergence** : Après ~25k steps
2. **Contacts** : Détection de contacts avec le cube
3. **Grasping** : Assistance automatique quand 2+ doigts touchent
4. **Stabilité** : Pas d'explosions de vitesse
5. **Progression** : Rewards en augmentation

## 💡 Conseils

1. **Commencez petit** : 50-100k steps pour tester
2. **Surveillez les logs** : Episode rewards toutes les 10 episodes  
3. **Patience** : Le grasping robotique demande du temps
4. **Ajustements** : Modifiez progressivement les hyperparamètres

## 🤝 Comparaison avec le collègue

| Fonctionnalité | Collègue | Notre version |
|---|---|---|
| Algorithme | TD3 ✅ | TD3 ✅ |
| Récompenses | Équilibrées ✅ | Équilibrées ✅ |
| Assistance grasping | ✅ | ✅ |
| Vidéos automatiques | ✅ | ✅ |
| Curriculum learning | ❌ | ❌ (simplifié) |
| Robustesse NaN/Inf | Basique | Avancée ✅ |
| Mode headless | ❌ | ✅ |

## 📞 Support

En cas de problème :

1. Vérifiez que toutes les dépendances sont installées
2. Testez avec `start_training.py` d'abord
3. Consultez les logs dans `simple_td3_results/`
4. Réduisez `total_timesteps` si nécessaire

---

**🎉 Bon entraînement !**