# 🔧 CORRECTIONS DES PROBLÈMES D'ENTRAÎNEMENT G1

## 🚨 **Problèmes Identifiés et Solutions**

### 1. **Instabilité MuJoCo (NaN/Inf dans QACC)**

**Problème**: Valeurs NaN/Infinies dans les accélérations des joints causant l'instabilité de simulation.

**Solutions implémentées**:
- ✅ **Environnement stabilisé** (`envs/stable_grasp_env.py`)
- ✅ **Paramètres MuJoCo optimisés** (timestep, solveur, tolérances)
- ✅ **Lissage des actions** pour éviter les changements brusques
- ✅ **Détection et gestion des instabilités**
- ✅ **Actions limitées** ([-0.5, 0.5] au lieu de [-1, 1])
- ✅ **Reset sécurisé** avec vérifications

### 2. **Épisodes Trop Courts (1 step)**

**Problème**: Simulation qui crash immédiatement.

**Solutions implémentées**:
- ✅ **Initialisation progressive** avec 100 étapes de stabilisation
- ✅ **Phase d'exploration conservative** (100 premiers épisodes)
- ✅ **Actions réduites** au début de l'entraînement
- ✅ **Gestion des échecs consécutifs**

### 3. **Pas d'Apprentissage (Récompense constante)**

**Problème**: Agent qui n'apprend pas, récompenses stagnantes.

**Solutions implémentées**:
- ✅ **Système de récompenses équilibré**
- ✅ **Récompense de stabilité** pour encourager les simulations valides
- ✅ **Entraînement moins fréquent** au début
- ✅ **Paramètres SAC conservateurs**

### 4. **Erreur FFmpeg pour Vidéos**

**Problème**: Backend manquant pour créer des vidéos MP4.

**Solutions implémentées**:
- ✅ **Enregistreur vidéo alternatif** (format pickle)
- ✅ **Détection automatique** de la disponibilité FFmpeg
- ✅ **Script de correction** des dépendances
- ✅ **Métadonnées détaillées** sauvegardées

## 🚀 **UTILISATION DU SYSTÈME CORRIGÉ**

### **Étape 1: Correction des Dépendances**

```bash
# Lancer le script de correction automatique
python3 fix_dependencies.py
```

### **Étape 2: Test du Système**

```bash
# Tester que tout fonctionne
python3 test_simple_grasp_basic.py
```

### **Étape 3: Entraînement Stabilisé**

```bash
# Entraînement court pour tester
python3 train_stable_grasp.py --episodes 50

# Entraînement normal
python3 train_stable_grasp.py --episodes 500

# Mode ultra-stable
python3 train_stable_grasp.py --episodes 300 --stable
```

## 📊 **Nouveaux Paramètres Stabilisés**

### **Environnement (`StableGraspEnv`)**
```python
- max_episode_steps: 200 (au lieu de 500)
- action_space: [-0.5, 0.5] (au lieu de [-1, 1])
- timestep: 0.005 (au lieu de 0.002)
- action_smoothing: 0.1 (lissage)
- iterations: 50 (solveur MuJoCo)
```

### **Agent SAC**
```python
- learning_rate: 1e-4 (au lieu de 3e-4)
- batch_size: 64 (au lieu de 256)
- buffer_size: 10000 (au lieu de 100000)
- hidden_sizes: [128, 128] (au lieu de [256, 256])
- training_frequency: 5 (entraîner moins souvent)
```

### **Récompenses Ajustées**
```python
- Récompense de base: +0.1 (pour rester stable)
- Contact détecté: +0.5 max (au lieu de 1.0)
- Cube soulevé: +2.0 max (au lieu de 10.0)
- Pénalité mouvement: -0.001 (au lieu de -0.01)
- Récompense stabilité: +0.1 ou -1.0
```

## 🎯 **Résultats Attendus**

### **Avant Corrections**
```
❌ Instabilité: WARNING: Nan, Inf in QACC
❌ Épisodes: 1 step seulement
❌ Récompenses: 0.50 constant
❌ Apprentissage: Aucun progrès
❌ Vidéos: Erreur FFmpeg
```

### **Après Corrections**
```
✅ Stabilité: Pas de warnings MuJoCo
✅ Épisodes: 50-200 steps normalement
✅ Récompenses: Progression visible
✅ Apprentissage: Métriques qui évoluent
✅ Vidéos: Format alternatif fonctionnel
```

## 📈 **Métriques à Surveiller**

### **Signes de Succès**
- 🟢 **Longueur épisodes**: > 20 steps
- 🟢 **Récompenses**: Croissance progressive
- 🟢 **Taux de contact**: > 10% après 100 épisodes
- 🟢 **Pas d'instabilité**: Aucun warning MuJoCo
- 🟢 **Buffer rempli**: > 1000 transitions

### **Signes de Problème**
- 🔴 **Épisodes courts**: < 5 steps
- 🔴 **Échecs consécutifs**: > 5
- 🔴 **Warnings MuJoCo**: NaN/Inf détectés
- 🔴 **Récompenses négatives**: < -5.0
- 🔴 **Pas de contact**: 0% après 200 épisodes

## 🛠️ **Diagnostic et Dépannage**

### **Si Instabilité Persiste**

1. **Vérifier le modèle G1**:
   ```bash
   # Valider le XML
   python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('results/g1_combined.xml'); print('✅ Modèle valide')"
   ```

2. **Réduire encore les paramètres**:
   ```python
   # Dans train_stable_grasp.py
   config['learning_rate'] = 5e-5  # Plus bas
   config['max_episode_steps'] = 100  # Plus court
   config['training_frequency'] = 20  # Moins fréquent
   ```

3. **Mode debug**:
   ```bash
   # Ajouter des prints de debug
   python3 train_stable_grasp.py --episodes 10 --lr 1e-5
   ```

### **Si Pas d'Apprentissage**

1. **Vérifier les capteurs**:
   ```python
   # Dans l'environnement, vérifier
   print(f"Capteurs de force: {len(self.force_sensor_ids)}")
   print(f"Position cube: {self.cube_body_id}")
   ```

2. **Augmenter les récompenses**:
   ```python
   # Dans _compute_reward()
   contact_reward *= 2.0  # Doubler
   height_reward *= 2.0   # Doubler
   ```

3. **Curriculum learning**:
   ```bash
   # Commencer plus simple
   python3 train_stable_grasp.py --episodes 1000
   ```

## 📁 **Structure des Fichiers Corrigés**

```
project/
├── envs/
│   ├── stable_grasp_env.py         # 🆕 Environnement stabilisé
│   └── simple_grasp_env.py         # Original (pour référence)
├── agents/
│   └── improved_sac_agent.py       # Agent SAC complet
├── utils/
│   ├── video_recorder.py           # Original
│   └── alternative_video_recorder.py # 🆕 Sans FFmpeg
├── train_stable_grasp.py           # 🆕 Script stabilisé
├── train_simple_grasp.py           # Original
├── fix_dependencies.py             # 🆕 Correction auto
└── README_FIXES.md                 # 🆕 Ce guide
```

## 🎮 **Commandes Essentielles**

```bash
# 1. CORRECTION AUTOMATIQUE
python3 fix_dependencies.py

# 2. TEST DU SYSTÈME
python3 test_simple_grasp_basic.py

# 3. ENTRAÎNEMENT RAPIDE (TEST)
python3 train_stable_grasp.py --episodes 25

# 4. ENTRAÎNEMENT NORMAL
python3 train_stable_grasp.py --episodes 200

# 5. ENTRAÎNEMENT ULTRA-STABLE
python3 train_stable_grasp.py --episodes 100 --stable --lr 5e-5
```

## 🔍 **Logs Importants**

### **Logs de Succès**
```
✅ Environnement stabilisé initialisé
✅ Modèle chargé: results/g1_combined.xml
🔧 Configuration des paramètres de stabilité...
   ✅ Paramètres de stabilité configurés
🚀 DÉBUT DE L'ENTRAÎNEMENT STABILISÉ

📊 Épisode 25/100
   Récompense: 0.85 ± 0.12
   Longueur: 45.2
   Succès: 5.0%
   Contact: 15.0%
   Échecs consécutifs: 0
   Buffer: 1250
```

### **Logs de Problème**
```
⚠️  Instabilité détectée épisode 15: simulation_unstable
🛑 Trop d'échecs consécutifs, arrêt de l'entraînement
❌ Erreur durant l'épisode 23: [détails]
```

## 💡 **Conseils d'Optimisation**

1. **Commencez petit**: 25-50 épisodes pour tester
2. **Surveillez les logs**: Arrêtez si échecs répétés
3. **Ajustez graduellement**: Paramètres par petits pas
4. **Vérifiez le modèle**: G1 XML doit être valide
5. **Patience**: L'apprentissage peut prendre du temps

---

**Version**: 2.0 Stabilisée  
**Date**: Août 2025  
**Statut**: ✅ Testé et Fonctionnel