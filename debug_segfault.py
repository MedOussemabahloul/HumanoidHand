#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour identifier la cause du segmentation fault
"""

import os
import sys
import traceback
import numpy as np
import torch
import mujoco

def test_imports():
    """Test des imports de base"""
    print("🔍 Test des imports...")
    try:
        import numpy as np
        print("✅ NumPy OK")
    except Exception as e:
        print(f"❌ NumPy ERROR: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch OK - Version: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ PyTorch ERROR: {e}")
        return False
    
    try:
        import mujoco
        print(f"✅ MuJoCo OK - Version: {mujoco.__version__}")
    except Exception as e:
        print(f"❌ MuJoCo ERROR: {e}")
        return False
    
    return True

def test_mujoco_basic():
    """Test basique de MuJoCo"""
    print("\n🔍 Test MuJoCo basique...")
    try:
        # Créer un modèle simple
        xml = """
        <mujoco>
            <worldbody>
                <body name="box" pos="0 0 0">
                    <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        print("✅ MuJoCo modèle simple créé")
        
        # Test forward
        mujoco.mj_forward(model, data)
        print("✅ MuJoCo forward OK")
        
        # Test step
        mujoco.mj_step(model, data)
        print("✅ MuJoCo step OK")
        
        return True
    except Exception as e:
        print(f"❌ MuJoCo ERROR: {e}")
        traceback.print_exc()
        return False

def test_torch_device():
    """Test du device PyTorch"""
    print("\n🔍 Test PyTorch device...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device sélectionné: {device}")
        
        # Test tensor sur device
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print("✅ Opérations tensorielles OK")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch device ERROR: {e}")
        traceback.print_exc()
        return False

def test_memory():
    """Test de la mémoire disponible"""
    print("\n🔍 Test mémoire...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ Mémoire totale: {memory.total / 1024**3:.1f} GB")
        print(f"✅ Mémoire disponible: {memory.available / 1024**3:.1f} GB")
        print(f"✅ Mémoire utilisée: {memory.percent:.1f}%")
        
        if memory.available < 2 * 1024**3:  # Moins de 2GB
            print("⚠️  Attention: Peu de mémoire disponible")
            return False
        return True
    except ImportError:
        print("⚠️  psutil non installé, impossible de vérifier la mémoire")
        return True
    except Exception as e:
        print(f"❌ Test mémoire ERROR: {e}")
        return False

def test_config_files():
    """Test des fichiers de configuration"""
    print("\n🔍 Test fichiers de configuration...")
    
    config_files = [
        "config/sac_grasp_lift.yaml",
        "config/default.yaml"
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} existe")
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   ✅ {config_file} parseable")
            except Exception as e:
                print(f"   ❌ {config_file} ERROR: {e}")
                all_good = False
        else:
            print(f"❌ {config_file} manquant")
            all_good = False
    
    return all_good

def test_assets():
    """Test des assets MuJoCo"""
    print("\n🔍 Test assets...")
    
    asset_dirs = ["assets", "envs"]
    all_good = True
    for asset_dir in asset_dirs:
        if os.path.exists(asset_dir):
            print(f"✅ {asset_dir}/ existe")
            files = os.listdir(asset_dir)
            xml_files = [f for f in files if f.endswith('.xml')]
            print(f"   {len(xml_files)} fichiers XML trouvés")
        else:
            print(f"❌ {asset_dir}/ manquant")
            all_good = False
    
    return all_good

def main():
    print("🚀 DIAGNOSTIC SEGMENTATION FAULT")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_mujoco_basic,
        test_torch_device,
        test_memory,
        test_config_files,
        test_assets
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} CRASH: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests réussis: {passed}/{total}")
    
    if passed == total:
        print("✅ Tous les tests passent - Le problème pourrait être dans le script principal")
    else:
        print("❌ Certains tests échouent - Vérifiez les erreurs ci-dessus")
    
    print("\n💡 RECOMMANDATIONS:")
    print("1. Vérifiez que tous les fichiers de configuration existent")
    print("2. Assurez-vous d'avoir suffisamment de mémoire (au moins 4GB)")
    print("3. Vérifiez que les fichiers XML MuJoCo sont valides")
    print("4. Essayez de lancer avec moins de threads: export OMP_NUM_THREADS=1")
    print("5. Vérifiez les logs système: dmesg | tail")

if __name__ == "__main__":
    main()