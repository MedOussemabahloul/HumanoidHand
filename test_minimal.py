#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test minimal pour diagnostiquer le segmentation fault
"""

import os
import sys
import gc

def test_basic_imports():
    """Test des imports de base"""
    print("🔍 Test des imports de base...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy non disponible: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch non disponible: {e}")
        return False
    
    try:
        import mujoco
        print(f"✅ MuJoCo: {mujoco.__version__}")
    except ImportError as e:
        print(f"❌ MuJoCo non disponible: {e}")
        return False
    
    return True

def test_mujoco_basic():
    """Test MuJoCo de base"""
    print("\n🧪 Test MuJoCo de base...")
    
    try:
        import mujoco
        
        # Créer un modèle minimal
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
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur MuJoCo: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_basic():
    """Test PyTorch de base"""
    print("\n🧪 Test PyTorch de base...")
    
    try:
        import torch
        import numpy as np
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Créer des tensors
        obs_dim = 10
        act_dim = 3
        
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur PyTorch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test de gestion mémoire"""
    print("\n🧪 Test de gestion mémoire...")
    
    try:
        import gc
        
        # Nettoyage avant
        gc.collect()
        print("✅ Nettoyage mémoire avant")
        
        # Test avec MuJoCo
        import mujoco
        
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="test">
  <worldbody>
    <body name="box" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>"""
        
        with open("test_model.xml", "w") as f:
            f.write(xml_content)
        
        # Créer et détruire plusieurs modèles
        for i in range(5):
            model = mujoco.MjModel.from_xml_path("test_model.xml")
            data = mujoco.MjData(model)
            
            # Simulation
            for j in range(5):
                mujoco.mj_step(model, data)
            
            # Nettoyage explicite
            del model, data
            gc.collect()
        
        os.remove("test_model.xml")
        print("✅ Test de gestion mémoire réussi")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur gestion mémoire: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    print("🚀 TEST MINIMAL MUJOCO + PYTORCH")
    print("=" * 40)
    
    # Test 1: Imports
    if not test_basic_imports():
        print("❌ Échec des imports de base")
        return False
    
    # Test 2: MuJoCo
    if not test_mujoco_basic():
        print("❌ Échec du test MuJoCo")
        return False
    
    # Test 3: PyTorch
    if not test_pytorch_basic():
        print("❌ Échec du test PyTorch")
        return False
    
    # Test 4: Gestion mémoire
    if not test_memory_management():
        print("❌ Échec du test de gestion mémoire")
        return False
    
    print("\n🎉 TOUS LES TESTS RÉUSSIS!")
    print("✅ Votre installation semble correcte")
    print("✅ Le problème de segmentation fault n'est pas lié aux dépendances de base")
    print("✅ Le problème est probablement dans votre code d'entraînement")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)