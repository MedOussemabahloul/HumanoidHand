# 🚀 GUIDE D'EXÉCUTION ULTRA-DÉTAILLÉ

## 📋 **ÉTAPE 1 : VÉRIFICATION SYSTÈME**

```bash
# Vérifier que vous êtes dans le bon répertoire
pwd
# Doit afficher le répertoire du projet avec scripts/, config/, etc.

# Vérifier système complet
python3 check_requirements.py
```

## 📦 **ÉTAPE 2 : INSTALLATION DÉPENDANCES**

### **Si vous avez un GPU NVIDIA :**
```bash
# PyTorch avec CUDA
pip install torch torchvision torchaudio

# Autres dépendances
pip install mujoco numpy scipy matplotlib PyYAML tensorboard
```

### **Si vous avez seulement un CPU :**
```bash
# PyTorch CPU seulement
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Autres dépendances
pip install mujoco numpy scipy matplotlib PyYAML tensorboard
```

### **Alternative avec conda :**
```bash
# GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU seulement
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Autres dépendances
conda install -c conda-forge mujoco numpy scipy matplotlib pyyaml tensorboard
```

## 🎯 **ÉTAPE 3 : CONFIGURATIONS DISPONIBLES**

### **🔧 Configuration `cpu` - OPTIMISÉE POUR VOTRE CAS**
- **Usage** : CPU seulement, optimisé performance
- **Durée** : 30-45 minutes
- **Steps** : 25,000 (réduit pour CPU)
- **Architecture** : [64, 64] (très léger)
- **Recommandé** : OUI pour CPU

### **⚡ Configuration `quick`**
- **Usage** : Tests rapides, validation
- **Durée** : 15-30 minutes 
- **Steps** : 50,000
- **Architecture** : [128, 128]
- **Recommandé** : Tests

### **🔧 Configuration `standard`**
- **Usage** : Entraînement équilibré
- **Durée** : 2-4h (plus long sur CPU)
- **Steps** : 2,000,000
- **Architecture** : [512, 512, 256]
- **Recommandé** : Si vous avez du temps

### **🏆 Configuration `production`**
- **Usage** : Haute performance finale
- **Durée** : 8-12h (très long sur CPU)
- **Steps** : 5,000,000
- **Architecture** : [768, 768, 512, 256]
- **Recommandé** : GPU seulement

## 🚀 **ÉTAPE 4 : EXÉCUTION**

### **🖥️ POUR CPU (RECOMMANDÉ POUR VOUS) :**

```bash
# 1. Test système rapide
python3 launch_training.py --config cpu --debug

# 2. Entraînement CPU optimisé complet
python3 launch_training.py --config cpu

# 3. Alternative directe
python3 scripts/train_sac_per_ultra.py --config config/train_config_cpu.yaml
```

### **🎮 POUR GPU (Si disponible) :**

```bash
# Test rapide
python3 launch_training.py --config quick

# Entraînement standard
python3 launch_training.py --config standard

# Production haute performance
python3 launch_training.py --config production --gpu 0
```

## 📊 **ÉTAPE 5 : MONITORING**

### **Logs en Temps Réel :**
```bash
# Terminal 1 : Lancer l'entraînement
python3 launch_training.py --config cpu

# Terminal 2 : Surveiller logs
tail -f outputs/cpu_optimized/logs/training.log

# Terminal 3 : Surveiller erreurs
tail -f outputs/cpu_optimized/logs/errors.log
```

### **TensorBoard (Si activé) :**
```bash
# Lancer TensorBoard
tensorboard --logdir runs/

# Accès web
http://localhost:6006
```

## 🔧 **OPTIMISATIONS CPU SPÉCIALES**

### **Variables d'Environnement CPU :**
```bash
# Optimiser threads CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Puis lancer entraînement
python3 launch_training.py --config cpu
```

### **Ajuster Selon Votre CPU :**

**Si CPU 4 cœurs :**
```bash
export OMP_NUM_THREADS=4
```

**Si CPU 8 cœurs :**
```bash
export OMP_NUM_THREADS=8
```

**Si CPU 16+ cœurs :**
```bash
export OMP_NUM_THREADS=12
```

## 🎯 **ÉTAPE 6 : COMMANDES COMPLÈTES**

### **🖥️ COMMANDE CPU COMPLÈTE (RECOMMANDÉE) :**

```bash
# Configuration optimale CPU
export OMP_NUM_THREADS=8
python3 launch_training.py --config cpu

# Ou version directe
python3 scripts/train_sac_per_ultra.py --config config/train_config_cpu.yaml --debug
```

### **🔄 REPRENDRE ENTRAÎNEMENT :**

```bash
# Reprendre depuis dernier checkpoint
python3 launch_training.py --config cpu --resume outputs/cpu_optimized/checkpoints/sac_per_best.pth

# Ou spécifier checkpoint exact
python3 launch_training.py --config cpu --resume outputs/cpu_optimized/checkpoints/sac_per_step_15000.pth
```

### **🐛 MODE DEBUG :**

```bash
# Mode debug avec validation renforcée
python3 launch_training.py --config cpu --debug

# Logs très détaillés
python3 scripts/train_sac_per_ultra.py --config config/train_config_cpu.yaml --debug
```

## 📈 **ÉTAPE 7 : SUIVI PROGRESSION**

### **Métriques Clés à Surveiller :**

```bash
# Récompenses moyennes (doit augmenter)
grep "Reward" outputs/cpu_optimized/logs/training.log | tail -10

# Taux de succès (doit augmenter vers 50%+)
grep "Success" outputs/cpu_optimized/logs/training.log | tail -5

# Erreurs (doit être vide)
cat outputs/cpu_optimized/logs/errors.log
```

### **Signes de Bon Entraînement :**
- ✅ **Reward moyen** augmente progressivement
- ✅ **Taux succès** passe de 0% à 30%+ 
- ✅ **Logs réguliers** sans erreurs
- ✅ **Checkpoints** créés régulièrement

### **Signes de Problème :**
- ❌ **Reward stagnant** ou diminue
- ❌ **Erreurs fréquentes** dans logs
- ❌ **Crash** ou arrêt inattendu
- ❌ **Pas de checkpoints** créés

## 🔥 **COMMANDES ULTRA-RAPIDES**

### **🏃 TEST ULTRA-RAPIDE (5 minutes) :**
```bash
# Modifier temporairement config CPU
sed -i 's/total_steps: 25000/total_steps: 2000/' config/train_config_cpu.yaml
python3 launch_training.py --config cpu --debug
# Remettre original
sed -i 's/total_steps: 2000/total_steps: 25000/' config/train_config_cpu.yaml
```

### **🚀 COMMANDE TOUT-EN-UN :**
```bash
# Vérification + installation + entraînement
python3 check_requirements.py && \
export OMP_NUM_THREADS=8 && \
python3 launch_training.py --config cpu
```

## 🆘 **RÉSOLUTION PROBLÈMES**

### **Erreur "command not found" :**
```bash
# Utiliser python3 au lieu de python
python3 launch_training.py --config cpu
```

### **Erreur "module not found" :**
```bash
# Réinstaller dépendances
pip install --upgrade torch mujoco numpy scipy PyYAML
```

### **Erreur "CUDA not available" :**
```bash
# Normal sur CPU, utiliser config CPU
python3 launch_training.py --config cpu
```

### **Entraînement trop lent :**
```bash
# Réduire architecture dans config/train_config_cpu.yaml
# Changer : hidden_sizes: [64, 64]
# En : hidden_sizes: [32, 32]
```

### **Manque de mémoire :**
```bash
# Réduire batch_size dans config/train_config_cpu.yaml
# Changer : batch_size: 32
# En : batch_size: 16
```

## ✅ **VALIDATION SUCCÈS**

### **L'entraînement réussit si :**

1. **Démarrage propre :**
```
🚀 ULTRA-ROBUST SAC + PER TRAINING SYSTEM
✅ UltraSACPERTrainer initialisé avec succès!
🚀 === DÉBUT ENTRAÎNEMENT ULTRA SAC+PER ===
```

2. **Progression visible :**
```
Ep   10 | Step    500 | R:   -5.23 | R̄₁₀:   -8.45 | Len: 150
Ep   20 | Step   1000 | R:   -3.12 | R̄₁₀:   -6.23 | Len: 145
Ep   30 | Step   1500 | R:    1.45 | R̄₁₀:   -2.18 | Len: 120
```

3. **Checkpoints créés :**
```
💾 Checkpoint: outputs/cpu_optimized/checkpoints/sac_per_step_5000.pth
💾 Checkpoint: outputs/cpu_optimized/checkpoints/sac_per_best.pth
```

4. **Fin propre :**
```
✅ Entraînement terminé en 1245.2s
📊 45 épisodes, 20.1 steps/sec
💾 Checkpoint: outputs/cpu_optimized/checkpoints/sac_per_final.pth
```

## 🎯 **RÉSUMÉ POUR VOUS (CPU)**

### **COMMANDE RECOMMANDÉE :**
```bash
python3 launch_training.py --config cpu
```

### **TEMPS ATTENDU :** 30-45 minutes
### **RÉSULTAT :** Modèle G1 entraîné pour grasp & lift
### **FICHIERS :** Modèles dans `outputs/cpu_optimized/checkpoints/`

**Vous pouvez exécuter sur CPU ! La config `cpu` est spécialement optimisée pour cela.** 🖥️✅