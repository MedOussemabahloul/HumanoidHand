# 🔧 Guide de Résolution du Segmentation Fault

## 🚨 Problème Identifié

Vous rencontrez un **segmentation fault (core dumped)** lors de l'exécution de votre script d'entraînement SAC. Ce problème peut avoir plusieurs causes, mais les plus courantes sont :

1. **Gestion mémoire défaillante** avec MuJoCo
2. **Problèmes de tensors PyTorch** (NaN, Inf, dimensions incorrectes)
3. **Configuration trop lourde** pour votre système
4. **Conflits entre CPU et GPU**
5. **Fichiers XML MuJoCo corrompus**

## 🔍 Diagnostic

### Étape 1: Test de Base
```bash
python3 test_minimal.py
```

Ce script teste :
- ✅ Imports de base (NumPy, PyTorch, MuJoCo)
- ✅ Chargement MuJoCo minimal
- ✅ Création de tensors PyTorch
- ✅ Gestion mémoire

### Étape 2: Vérification Système
```bash
python3 debug_segmentation_fault.py
```

Ce script fournit :
- 📊 Informations système détaillées
- 🧪 Tests de gestion mémoire
- 📝 Script de test minimal

## 🛠️ Solutions

### Solution 1: Script Sécurisé (Recommandé)

Utilisez la version sécurisée que j'ai créée :

```bash
python3 launch_training_safe.py --config quick
```

**Améliorations apportées :**
- ✅ Gestion mémoire améliorée
- ✅ Validation des données
- ✅ Gestion des erreurs robuste
- ✅ Configuration sécurisée
- ✅ Gradient clipping
- ✅ Nettoyage mémoire automatique

### Solution 2: Configuration Réduite

Si le problème persiste, utilisez une configuration encore plus légère :

```yaml
# config/train_config_minimal.yaml
task:
  max_steps_per_episode: 50
  output_dir: "results"
  
rl:
  batch_size: 16          # Réduit de 32
  replay_size: 5000       # Réduit de 10000
  total_steps: 500        # Réduit de 1000
  num_updates: 1          # Réduit
  hidden_size: 128        # Réduit de 256
```

### Solution 3: Corrections Manuelles

Si vous voulez corriger votre code original, appliquez ces modifications :

#### 1. Gestion Mémoire MuJoCo
```python
# Dans votre classe SACTrainer.__init__
import gc

# Après le chargement MuJoCo
gc.collect()
self.model = mujoco.MjModel.from_xml_path(model_xml)
self.data = mujoco.MjData(self.model)
gc.collect()
```

#### 2. Validation des Tensors
```python
def safe_tensor_creation(data, device, dtype=torch.float32):
    if data is None:
        raise ValueError("Données None")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Vérification des valeurs NaN/Inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("⚠️  ATTENTION: Données NaN ou Inf détectées")
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    tensor = torch.as_tensor(data, device=device, dtype=dtype)
    
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("Tensor contient des valeurs NaN ou Inf")
    
    return tensor
```

#### 3. Gradient Clipping
```python
# Dans votre méthode update()
torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
```

#### 4. Gestion des Erreurs
```python
try:
    # Votre code d'entraînement
    pass
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    # Nettoyage mémoire
    gc.collect()
    raise
```

## 🔧 Corrections Spécifiques

### Problème CPU vs GPU

Si vous utilisez CPU et que le problème persiste :

1. **Forcer l'utilisation CPU :**
```python
self.device = torch.device("cpu")
torch.set_num_threads(1)  # Limiter les threads
```

2. **Réduire la complexité :**
```python
# Réseaux plus simples
self.hidden = 128  # Au lieu de 256 ou plus
self.batch_size = 16  # Au lieu de 32 ou plus
```

### Problème MuJoCo

1. **Vérifier les fichiers XML :**
```bash
# Vérifier la syntaxe XML
xmllint --noout your_model.xml
```

2. **Créer des modèles minimaux :**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="minimal">
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
```

## 📋 Checklist de Résolution

- [ ] Exécuter `test_minimal.py` - Vérifier que les dépendances fonctionnent
- [ ] Utiliser `launch_training_safe.py` - Script sécurisé
- [ ] Réduire la configuration si nécessaire
- [ ] Vérifier les fichiers XML MuJoCo
- [ ] Ajouter la gestion d'erreurs
- [ ] Implémenter le gradient clipping
- [ ] Valider les tensors

## 🚀 Utilisation du Script Sécurisé

```bash
# Test rapide
python3 launch_training_safe.py --config quick

# Test moyen
python3 launch_training_safe.py --config medium

# Entraînement complet
python3 launch_training_safe.py --config full
```

## 📊 Monitoring

Le script sécurisé inclut :
- 📈 Logging TensorBoard automatique
- 🔍 Validation des données en temps réel
- 💾 Checkpoints automatiques
- ⚠️ Gestion des erreurs détaillée

## 🆘 Si le Problème Persiste

1. **Vérifiez la mémoire système :**
```bash
free -h
```

2. **Vérifiez les logs système :**
```bash
dmesg | tail -20
```

3. **Testez avec un environnement virtuel propre :**
```bash
python3 -m venv test_env
source test_env/bin/activate
pip install torch mujoco numpy pyyaml
```

4. **Contactez le support avec :**
- Sortie de `test_minimal.py`
- Sortie de `debug_segmentation_fault.py`
- Configuration utilisée
- Informations système

## ✅ Résultat Attendu

Avec les corrections, vous devriez voir :
```
🚀 ULTRA-ROBUST SAC PER TRAINING SYSTEM (VERSION SÉCURISÉE)
============================================================
✅ Trainer SAC initialisé avec succès
🚀 Début de l'entraînement...
Step 0/1000, Reward: -0.123
Step 100/1000, Reward: 0.456
...
✅ Entraînement terminé. Policy sauvegardée: results/policy_final.pth
🎉 Entraînement terminé avec succès!
```

---

**Note :** Le segmentation fault n'est pas forcément lié à l'utilisation du CPU, mais plutôt à la gestion mémoire et à la validation des données. Les corrections apportées devraient résoudre le problème dans la plupart des cas.