#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic et correction du segmentation fault
Problèmes identifiés et solutions :
1. Gestion mémoire MuJoCo
2. Problèmes de device CPU/GPU
3. Gestion des tensors PyTorch
4. Validation des données d'entrée
"""

import os
import sys
import gc
import traceback
import numpy as np
import torch
import mujoco
from contextlib import contextmanager

# Ajouter le répertoire parent au path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_system_info():
    """Vérifie les informations système et les dépendances"""
    print("🔍 DIAGNOSTIC SYSTÈME")
    print("=" * 50)
    
    # Version Python
    print(f"Python version: {sys.version}")
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # MuJoCo
    try:
        print(f"MuJoCo version: {mujoco.__version__}")
    except:
        print("MuJoCo version: Non disponible")
    
    # Mémoire disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Mémoire totale: {memory.total / (1024**3):.2f} GB")
        print(f"Mémoire disponible: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil non installé - impossible de vérifier la mémoire")
    
    print()

def safe_tensor_creation(data, device, dtype=torch.float32):
    """Crée un tensor de manière sécurisée avec validation"""
    try:
        # Validation des données
        if data is None:
            raise ValueError("Données None")
        
        # Conversion en numpy si nécessaire
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Vérification des valeurs NaN/Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("⚠️  ATTENTION: Données NaN ou Inf détectées")
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Création du tensor
        tensor = torch.as_tensor(data, device=device, dtype=dtype)
        
        # Vérification finale
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contient des valeurs NaN ou Inf")
        
        return tensor
    
    except Exception as e:
        print(f"❌ Erreur lors de la création du tensor: {e}")
        print(f"   Données: shape={getattr(data, 'shape', 'N/A')}, dtype={getattr(data, 'dtype', 'N/A')}")
        raise

@contextmanager
def safe_mujoco_context():
    """Contexte sécurisé pour les opérations MuJoCo"""
    try:
        # Nettoyage mémoire avant
        gc.collect()
        yield
    except Exception as e:
        print(f"❌ Erreur MuJoCo: {e}")
        traceback.print_exc()
        raise
    finally:
        # Nettoyage mémoire après
        gc.collect()

def validate_model_xml(xml_path):
    """Valide le fichier XML MuJoCo"""
    print(f"🔍 Validation du fichier XML: {xml_path}")
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Fichier XML introuvable: {xml_path}")
    
    try:
        # Test de chargement MuJoCo
        with safe_mujoco_context():
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            
            print(f"✅ XML valide - nq={model.nq}, nv={model.nv}, nu={model.nu}")
            return model, data
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement XML: {e}")
        raise

def create_safe_trainer_config():
    """Crée une configuration sécurisée pour l'entraînement"""
    return {
        "task": {
            "cube_body_name": "cube",
            "max_steps_per_episode": 100,  # Réduit pour les tests
            "touch_sensors": [],
            "force_sensors": [],
            "include_orientation_reward": False,
            "force_reward_weight_normal": 0.0,
            "force_reward_weight_tangential": 0.0,
            "translation_penalty_weight": 0.0,
            "output_dir": "results",
            "save_freq_steps": 1000
        },
        "rl": {
            "gamma": 0.99,
            "alpha": 0.2,
            "learning_rate": 3e-4,
            "hidden_size": 256,
            "batch_size": 32,  # Réduit pour éviter les problèmes mémoire
            "replay_size": 10000,  # Réduit
            "start_steps": 100,
            "update_after": 100,
            "update_every": 1,
            "num_updates": 1,  # Réduit
            "total_steps": 1000,  # Réduit pour les tests
            "tau": 0.005,
            "act_limit": 1.0
        }
    }

def test_memory_management():
    """Test de gestion mémoire"""
    print("🧪 TEST DE GESTION MÉMOIRE")
    print("=" * 30)
    
    try:
        # Test 1: Création de tensors
        print("Test 1: Création de tensors...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tensors de test
        test_data = np.random.randn(100, 50).astype(np.float32)
        tensor = safe_tensor_creation(test_data, device)
        print(f"✅ Tensor créé: {tensor.shape} sur {device}")
        
        # Test 2: Opérations PyTorch
        print("Test 2: Opérations PyTorch...")
        result = torch.matmul(tensor, tensor.T)
        print(f"✅ Matrice multiplication: {result.shape}")
        
        # Test 3: Nettoyage mémoire
        print("Test 3: Nettoyage mémoire...")
        del tensor, result
        gc.collect()
        print("✅ Nettoyage mémoire réussi")
        
    except Exception as e:
        print(f"❌ Erreur dans le test mémoire: {e}")
        traceback.print_exc()

def create_minimal_test_script():
    """Crée un script de test minimal pour identifier le problème"""
    print("📝 CRÉATION DU SCRIPT DE TEST MINIMAL")
    print("=" * 40)
    
    script_content = '''#!/usr/bin/env python3
import os
import sys
import gc
import numpy as np
import torch
import mujoco

# Configuration minimale
def test_minimal_setup():
    print("🚀 Test minimal MuJoCo + PyTorch")
    
    # 1. Test PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Test MuJoCo simple
    try:
        # Créer un modèle MuJoCo minimal
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="test">
  <worldbody>
    <body name="box" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>"""
        
        # Sauvegarder temporairement
        with open("test_model.xml", "w") as f:
            f.write(xml_content)
        
        # Charger MuJoCo
        model = mujoco.MjModel.from_xml_path("test_model.xml")
        data = mujoco.MjData(model)
        
        print(f"✅ MuJoCo chargé: nq={model.nq}, nv={model.nv}")
        
        # Test simulation
        for i in range(10):
            mujoco.mj_step(model, data)
        
        print("✅ Simulation MuJoCo réussie")
        
        # Nettoyage
        os.remove("test_model.xml")
        
    except Exception as e:
        print(f"❌ Erreur MuJoCo: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Test PyTorch + MuJoCo ensemble
    try:
        # Créer des tensors avec des données simulées
        obs_dim = 10
        act_dim = 3
        
        # Observations simulées
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        
        # Actions simulées
        action = np.random.randn(act_dim).astype(np.float32)
        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
        
        print(f"✅ Tensors créés: obs={obs_tensor.shape}, action={action_tensor.shape}")
        
        # Test réseau simple
        class SimpleNet(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                return self.fc(x)
        
        net = SimpleNet(obs_dim, act_dim).to(device)
        output = net(obs_tensor)
        print(f"✅ Réseau testé: output={output.shape}")
        
    except Exception as e:
        print(f"❌ Erreur PyTorch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_setup()
'''
    
    with open("test_minimal.py", "w") as f:
        f.write(script_content)
    
    print("✅ Script de test minimal créé: test_minimal.py")
    print("   Exécutez: python test_minimal.py")

def main():
    """Fonction principale de diagnostic"""
    print("🔧 DIAGNOSTIC SEGMENTATION FAULT")
    print("=" * 50)
    
    # 1. Informations système
    check_system_info()
    
    # 2. Test gestion mémoire
    test_memory_management()
    
    # 3. Créer script de test minimal
    create_minimal_test_script()
    
    print("\n📋 RECOMMANDATIONS:")
    print("=" * 30)
    print("1. Exécutez: python test_minimal.py")
    print("2. Si le test minimal échoue, le problème est dans l'installation")
    print("3. Si le test minimal réussit, le problème est dans votre code")
    print("4. Vérifiez les fichiers XML MuJoCo")
    print("5. Réduisez la taille du batch et du replay buffer")
    print("6. Utilisez torch.no_grad() pour les inférences")
    print("7. Ajoutez des try/except autour des opérations critiques")

if __name__ == "__main__":
    main()