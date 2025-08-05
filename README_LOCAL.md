# Système d'Entraînement G1 - Installation Locale

## 🚀 Installation Rapide

1. **Vérifier le système:**
   ```bash
   python3 test_simple_grasp_basic.py
   ```

2. **Installer les dépendances:**
   ```bash
   pip install numpy torch gymnasium mujoco matplotlib imageio
   ```

3. **Placer le modèle:**
   - Copiez votre fichier `g1_combined.xml` dans le dossier `results/`

4. **Lancer l'entraînement:**
   ```bash
   python3 train_simple_grasp.py --episodes 100
   ```

## 📁 Structure Créée

```
./
├── envs/
│   └── simple_grasp_env.py     # Environnement de saisie
├── agents/
│   └── improved_sac_agent.py   # Agent SAC 
├── utils/
│   └── video_recorder.py       # Enregistrement vidéo
├── results/
│   └── g1_combined.xml         # [À placer] Modèle MuJoCo
├── training_results/           # Résultats d'entraînement
│   ├── models/                 # Modèles sauvegardés
│   ├── videos/                 # Vidéos d'épisodes
│   └── logs/                   # Métriques JSON
├── train_simple_grasp.py       # Script principal
└── test_simple_grasp_basic.py  # Tests de validation
```

## 🎯 Fonctionnalités

- ✅ **Détection de contact** via capteurs de force
- ✅ **Phases automatiques**: approche → contact → saisie → levage  
- ✅ **Agent SAC** avec replay buffer et target networks
- ✅ **Curriculum learning** adaptatif
- ✅ **Enregistrement vidéo** automatique
- ✅ **Métriques détaillées** et sauvegarde

## 🔧 Utilisation

### Test du Système
```bash
python3 test_simple_grasp_basic.py
```

### Entraînement Court
```bash
python3 train_simple_grasp.py --episodes 50
```

### Entraînement Complet
```bash
python3 train_simple_grasp.py --episodes 1000 --lr 3e-4
```

## 📊 Résultats

Les résultats sont sauvegardés dans `training_results/`:
- **Modèles**: `models/final_model.pth`
- **Vidéos**: `videos/episode_*.mp4`
- **Métriques**: `logs/final_metrics.json`

## 🆘 Dépannage

1. **Erreur "Module not found"**: Installer les dépendances Python
2. **Erreur "Model not found"**: Placer g1_combined.xml dans results/
3. **Pas de vidéos**: Installer imageio (`pip install imageio`)
4. **Erreurs MuJoCo**: Vérifier que le modèle XML est valide

## 🎮 Système de Récompenses

- **Contact détecté**: +0.5 à +1.0 (proportionnel à la force)
- **Cube soulevé**: +20 × hauteur (max +10.0)
- **Mouvement excessif**: -0.01 × énergie_action
- **Cube tombé**: -5.0

Le robot apprend progressivement à:
1. Approcher le cube avec ses mains
2. Détecter le contact via les capteurs de force
3. Fermer les doigts pour saisir
4. Soulever le cube avec succès

---
**Système créé automatiquement**
