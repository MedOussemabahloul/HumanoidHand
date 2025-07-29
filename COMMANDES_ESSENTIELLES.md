# 🚀 COMMANDES ESSENTIELLES - SAC+PER ULTRA

## 📋 **SÉQUENCE COMPLÈTE D'EXÉCUTION**

### **1️⃣ INSTALLATION AUTOMATIQUE**
```bash
# Installation automatique de toutes les dépendances
python3 install_dependencies.py
```

### **2️⃣ VÉRIFICATION SYSTÈME**
```bash
# Vérifier que tout est prêt
python3 check_requirements.py
```

### **3️⃣ ENTRAÎNEMENT (CHOISIR SELON VOTRE MATÉRIEL)**

#### **🖥️ CPU (RECOMMANDÉ POUR VOUS)**
```bash
# Entraînement optimisé CPU (30-45 min)
python3 launch_training.py --config cpu

# Avec optimisations threads
export OMP_NUM_THREADS=8
python3 launch_training.py --config cpu
```

#### **⚡ TESTS RAPIDES**
```bash
# Test rapide (15-30 min)
python3 launch_training.py --config quick --debug

# Test ultra-rapide (5 min)
python3 launch_training.py --config cpu --debug
```

#### **🎮 GPU (Si disponible)**
```bash
# Standard (2-3h)
python3 launch_training.py --config standard

# Production (8-12h)
python3 launch_training.py --config production --gpu 0
```

## 📊 **MONITORING**

### **Logs Temps Réel**
```bash
# Surveiller entraînement
tail -f outputs/cpu_optimized/logs/training.log

# Surveiller erreurs
tail -f outputs/cpu_optimized/logs/errors.log
```

### **TensorBoard**
```bash
# Lancer monitoring visuel
tensorboard --logdir runs/
# Puis http://localhost:6006
```

## 🔄 **GESTION ENTRAÎNEMENT**

### **Reprendre Entraînement**
```bash
# Reprendre depuis meilleur checkpoint
python3 launch_training.py --config cpu --resume outputs/cpu_optimized/checkpoints/sac_per_best.pth
```

### **Arrêter Proprement**
```bash
# Ctrl+C pour arrêt propre avec sauvegarde
```

## 🎯 **VOS CONFIGURATIONS**

### **📁 Fichiers Utilisés**
- **Script** : `scripts/train_sac_per_ultra.py` ✅
- **Task** : `tasks/grasp/grasp_lift_task_optimized.py` ✅
- **Quaternions** : `tasks/planner/high_level_planner.py` ✅
- **Modèle** : `results/g1_combined.xml` ✅

### **🔧 Configurations Disponibles**
1. **`cpu`** - Optimisée CPU (25K steps, [64,64]) 🖥️
2. **`quick`** - Tests rapides (50K steps, [128,128]) ⚡
3. **`standard`** - Équilibrée (2M steps, [512,512,256]) 🔧
4. **`production`** - Haute performance (5M steps, [768,768,512,256]) 🏆

## 🔥 **COMMANDE RECOMMANDÉE POUR VOUS**

```bash
# COMMANDE UNIQUE TOUT-EN-UN (CPU)
export OMP_NUM_THREADS=8 && python3 launch_training.py --config cpu
```

### **Résultat Attendu :**
- ⏱️ **Durée** : 30-45 minutes
- 📈 **Steps** : 25,000 
- 🎯 **Modèle** : `outputs/cpu_optimized/checkpoints/sac_per_final.pth`
- 📊 **Reward** : De -10 vers +5-10
- ✅ **Succès** : 30-70% des épisodes

## 🆘 **DÉPANNAGE EXPRESS**

### **Erreur Dépendances**
```bash
pip install torch torchvision torchaudio mujoco numpy scipy PyYAML
```

### **Erreur "command not found"**
```bash
# Remplacer 'python' par 'python3'
python3 launch_training.py --config cpu
```

### **Entraînement Lent**
```bash
# Réduire dans config/train_config_cpu.yaml:
# hidden_sizes: [32, 32]
# batch_size: 16
# total_steps: 10000
```

### **Manque Mémoire**
```bash
# Réduire batch_size dans config
# Fermer autres applications
export OMP_NUM_THREADS=4
```

## ✅ **VALIDATION SUCCÈS**

### **Démarrage OK**
```
🚀 ULTRA-ROBUST SAC + PER TRAINING SYSTEM
✅ UltraSACPERTrainer initialisé avec succès!
```

### **Progression OK**
```
Ep   10 | Step    500 | R:   -5.23 | R̄₁₀:   -8.45 | Len: 150
Ep   20 | Step   1000 | R:   -3.12 | R̄₁₀:   -6.23 | Len: 145
Ep   30 | Step   1500 | R:    1.45 | R̄₁₀:   -2.18 | Len: 120
```

### **Fin OK**
```
✅ Entraînement terminé en 1245.2s
💾 Checkpoint: outputs/cpu_optimized/checkpoints/sac_per_final.pth
```

---

## 🎯 **RÉSUMÉ ULTRA-SIMPLE**

**VOUS POUVEZ EXÉCUTER SUR CPU !** 

**Commande magique :**
```bash
python3 launch_training.py --config cpu
```

**C'est tout !** Le système fait le reste automatiquement. 🚀✅