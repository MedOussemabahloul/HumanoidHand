#!/usr/bin/env python3
"""
🔍 VÉRIFICATION SYSTÈME POUR SAC+PER ULTRA
=========================================

Script de vérification des dépendances et configuration
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Vérifier version Python"""
    print("🐍 PYTHON VERSION")
    print(f"   Version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("   ❌ ERREUR: Python 3.8+ requis")
        return False
    else:
        print("   ✅ Version OK")
        return True

def check_dependencies():
    """Vérifier dépendances Python"""
    print("\n📦 DÉPENDANCES PYTHON")
    
    required_packages = {
        'numpy': 'NumPy pour calculs scientifiques',
        'torch': 'PyTorch pour réseaux neuronaux',
        'mujoco': 'MuJoCo pour simulation physique',
        'yaml': 'PyYAML pour configurations',
        'scipy': 'SciPy pour fonctions scientifiques',
        'matplotlib': 'Matplotlib pour visualisation (optionnel)',
        'tensorboard': 'TensorBoard pour monitoring (optionnel)'
    }
    
    missing = []
    available = []
    
    for package, description in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'inconnue')
            print(f"   ✅ {package} ({version}) - {description}")
            available.append(package)
        except ImportError:
            print(f"   ❌ {package} - {description}")
            missing.append(package)
    
    return missing, available

def check_cuda():
    """Vérifier disponibilité CUDA"""
    print("\n🎮 CUDA / GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Inconnu"
            print(f"   ✅ CUDA disponible")
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ Nombre GPUs: {gpu_count}")
            
            # Test allocation mémoire
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print(f"   ✅ Test allocation GPU: OK")
                return True
            except Exception as e:
                print(f"   ⚠️  Test allocation GPU: {e}")
                return False
        else:
            print("   ⚠️  CUDA non disponible - CPU seulement")
            return False
            
    except ImportError:
        print("   ❌ PyTorch non installé")
        return False

def check_files():
    """Vérifier fichiers système"""
    print("\n📁 FICHIERS SYSTÈME")
    
    required_files = {
        'scripts/train_sac_per_ultra.py': 'Script entraînement principal',
        'tasks/grasp/grasp_lift_task_optimized.py': 'Task grasp optimisée',
        'tasks/planner/high_level_planner.py': 'Planner quaternions',
        'results/g1_combined.xml': 'Modèle G1 combiné',
        'launch_training.py': 'Script lancement simple',
        'config/sac_grasp_lift.yaml': 'Config standard',
        'config/train_config_quick.yaml': 'Config test rapide',
        'config/train_config_production.yaml': 'Config production'
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes) - {description}")
        else:
            print(f"   ❌ {file_path} - {description}")
            missing_files.append(file_path)
    
    return missing_files

def check_mujoco():
    """Vérifier installation MuJoCo"""
    print("\n🔬 MUJOCO")
    
    try:
        import mujoco
        print(f"   ✅ MuJoCo version: {mujoco.__version__}")
        
        # Test chargement modèle
        try:
            model = mujoco.MjModel.from_xml_path("results/g1_combined.xml")
            data = mujoco.MjData(model)
            print(f"   ✅ Test modèle G1: OK")
            print(f"   ✅ DOF: {model.nv}, Actuateurs: {model.nu}")
            return True
        except Exception as e:
            print(f"   ❌ Test modèle G1: {e}")
            return False
            
    except ImportError:
        print("   ❌ MuJoCo non installé")
        return False

def estimate_performance():
    """Estimer performance système"""
    print("\n⚡ ESTIMATION PERFORMANCE")
    
    try:
        import torch
        import time
        
        # Test CPU
        start = time.time()
        x = torch.randn(1000, 1000)
        y = torch.mm(x, x)
        cpu_time = time.time() - start
        print(f"   💻 Test CPU (1000x1000 matrix): {cpu_time:.3f}s")
        
        if torch.cuda.is_available():
            # Test GPU
            start = time.time()
            x_gpu = torch.randn(1000, 1000).cuda()
            y_gpu = torch.mm(x_gpu, x_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"   🎮 Test GPU (1000x1000 matrix): {gpu_time:.3f}s")
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"   🚀 Accélération GPU: {speedup:.1f}x")
            
            return 'gpu'
        else:
            print("   💻 Mode CPU seulement")
            return 'cpu'
            
    except Exception as e:
        print(f"   ❌ Erreur test performance: {e}")
        return 'unknown'

def provide_recommendations(performance_mode, missing_deps, missing_files):
    """Fournir recommandations"""
    print("\n🎯 RECOMMANDATIONS")
    
    if missing_deps:
        print("   📦 INSTALLER DÉPENDANCES:")
        print("   pip install torch torchvision torchaudio")
        print("   pip install mujoco")
        print("   pip install numpy scipy matplotlib")
        print("   pip install PyYAML tensorboard")
    
    if missing_files:
        print("   📁 FICHIERS MANQUANTS:")
        for file in missing_files:
            print(f"      - {file}")
    
    print("\n🚀 CONFIGURATION RECOMMANDÉE:")
    
    if performance_mode == 'gpu':
        print("   ✅ GPU disponible - Utilisez config standard ou production")
        print("   📜 Commande: python launch_training.py --config standard")
        print("   ⏱️  Temps estimé: 2-3h pour config standard")
        
    elif performance_mode == 'cpu':
        print("   💻 CPU seulement - Utilisez config quick")
        print("   📜 Commande: python launch_training.py --config quick")
        print("   ⏱️  Temps estimé: 1-2h pour config quick (CPU plus lent)")
        print("   💡 CONSEIL: Réduisez total_steps à 10000 pour test rapide")
        
    else:
        print("   ⚠️  Configuration inconnue - Utilisez config quick")
    
    print("\n🔧 OPTIMISATIONS CPU:")
    if performance_mode == 'cpu':
        print("   - Augmentez num_threads dans config")
        print("   - Réduisez batch_size (64 au lieu de 256)")
        print("   - Réduisez hidden_sizes ([128, 128] au lieu de [512, 512, 256])")
        print("   - Réduisez replay_size (50000 au lieu de 1000000)")

def main():
    """Vérification complète du système"""
    
    print("🔍 VÉRIFICATION SYSTÈME SAC+PER ULTRA")
    print("="*50)
    
    # Vérifications
    python_ok = check_python_version()
    missing_deps, available_deps = check_dependencies()
    cuda_available = check_cuda()
    missing_files = check_files()
    mujoco_ok = check_mujoco()
    performance_mode = estimate_performance()
    
    # Résumé
    print("\n📊 RÉSUMÉ")
    print(f"   Python: {'✅' if python_ok else '❌'}")
    print(f"   Dépendances: {'✅' if not missing_deps else f'❌ {len(missing_deps)} manquantes'}")
    print(f"   CUDA: {'✅' if cuda_available else '⚠️  CPU seulement'}")
    print(f"   Fichiers: {'✅' if not missing_files else f'❌ {len(missing_files)} manquants'}")
    print(f"   MuJoCo: {'✅' if mujoco_ok else '❌'}")
    
    # Recommandations
    provide_recommendations(performance_mode, missing_deps, missing_files)
    
    # Status final
    ready = python_ok and not missing_deps and not missing_files and mujoco_ok
    
    print(f"\n🎯 STATUT: {'✅ PRÊT' if ready else '❌ CONFIGURATION REQUISE'}")
    
    return 0 if ready else 1

if __name__ == "__main__":
    exit(main())