# 🔧 RÉPARATION SEGMENTATION FAULT

## 💥 **PROBLÈME IDENTIFIÉ**

```
(venv) python scripts/train_sac_per_ultra.py --config config/sac_grasp_lift.yaml
Segmentation fault (core dumped)
```

## 🎯 **CAUSE : DÉPENDANCES MANQUANTES**

Le segfault est causé par l'absence des bibliothèques critiques :
- ❌ **numpy** - Calculs scientifiques
- ❌ **torch** - Réseaux neuronaux  
- ❌ **mujoco** - Simulation physique
- ❌ **scipy** - Fonctions scientifiques

## 🚀 **SOLUTION COMPLÈTE**

### **📦 ÉTAPE 1 : INSTALLATION DÉPENDANCES**

```bash
# Activez votre virtual env si pas déjà fait
source venv/bin/activate

# Installation PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installation MuJoCo
pip install mujoco

# Installation packages scientifiques
pip install numpy scipy matplotlib

# Installation configuration & monitoring
pip install PyYAML tensorboard
```

### **🔍 ÉTAPE 2 : VÉRIFICATION**

```bash
# Test imports un par un
python3 -c "import numpy; print('✅ NumPy OK')"
python3 -c "import torch; print('✅ PyTorch OK')"
python3 -c "import mujoco; print('✅ MuJoCo OK')"

# Test combiné
python3 -c "import torch, mujoco, numpy, yaml; print('✅ Tous imports OK')"

# Vérification système complète
python3 check_requirements.py
```

### **🚀 ÉTAPE 3 : TEST SÉCURISÉ**

```bash
# Test avec configuration CPU optimisée
python3 launch_training.py --config cpu --debug

# Si ça marche, essayez la config standard
python3 launch_training.py --config standard --debug
```

## 🆘 **SI PROBLÈME PERSISTE**

### **🔧 Réinstallation complète MuJoCo**
```bash
pip uninstall mujoco -y
pip install mujoco
python3 -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
```

### **🔬 Test modèle G1**
```bash
python3 -c "
import mujoco
try:
    model = mujoco.MjModel.from_xml_path('results/g1_combined.xml')
    print('✅ Modèle G1 charge OK')
except Exception as e:
    print(f'❌ Erreur modèle G1: {e}')
"
```

### **🐛 Debug avancé**
```bash
# Mode debug ultra-détaillé
export PYTHONFAULTHANDLER=1
export MUJOCO_GL=egl
python3 scripts/train_sac_per_ultra.py --config config/train_config_cpu.yaml --debug
```

## ✅ **VALIDATION SUCCÈS**

### **Après installation, vous devriez voir :**

```bash
$ python3 check_requirements.py
🔍 VÉRIFICATION SYSTÈME SAC+PER ULTRA
==================================================
🐍 PYTHON VERSION
   ✅ Version OK

📦 DÉPENDANCES PYTHON
   ✅ numpy (1.24.3) - NumPy pour calculs scientifiques
   ✅ torch (2.0.1) - PyTorch pour réseaux neuronaux
   ✅ mujoco (3.3.4) - MuJoCo pour simulation physique
   ✅ yaml (6.0.2) - PyYAML pour configurations
   ✅ scipy (1.11.1) - SciPy pour fonctions scientifiques

🎯 STATUT: ✅ PRÊT
```

### **Test final :**
```bash
$ python3 launch_training.py --config cpu --debug
🚀 ULTRA-ROBUST SAC + PER TRAINING SYSTEM
✅ UltraSACPERTrainer initialisé avec succès!
```

## 🎯 **COMMANDES FINALES RECOMMANDÉES**

### **🖥️ Pour CPU (sécurisé) :**
```bash
python3 launch_training.py --config cpu
```

### **🎮 Pour GPU (si disponible) :**
```bash
python3 launch_training.py --config standard
```

### **⚡ Test rapide :**
```bash
python3 launch_training.py --config cpu --debug
```

---

## 💡 **PRÉVENTION FUTURE**

### **Toujours vérifier avant utilisation :**
```bash
python3 check_requirements.py
```

### **Environnement virtuel recommandé :**
```bash
python3 -m venv venv
source venv/bin/activate
# Puis installer dépendances
```

**Le segfault est résolu avec l'installation des dépendances !** 🎉