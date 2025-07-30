
#!/usr/bin/env python3
"""
📦 INSTALLATION AUTOMATIQUE DÉPENDANCES
======================================

Script d'installation automatique pour SACPER Ultra
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Exécuter commande avec gestion erreurs"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               capture_output=True, text=True)
        print(f"   ✅ {description}: OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description}: ERREUR")
        print(f"   📜 Commande: {cmd}")
        print(f"   📜 Erreur: {e.stderr}")
        return False

def detect_system():
    """Détecter système et GPU"""
    print("🔍 DÉTECTION SYSTÈME")
    
    # Détecter CUDA
    has_cuda = False
    try:
        result = subprocess.run("nvidia-smi", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            has_cuda = True
            print("   ✅ GPU NVIDIA détecté")
        else:
            print("   ⚠️  Pas de GPU NVIDIA")
    except:
        print("   ⚠️  nvidia-smi non trouvé")
    
    # Détecter Python
    python_version = sys.version_info
    print(f"   ✅ Python {python_version.major}.{python_version.minor}")
    
    # Détecter pip/conda
    has_pip = subprocess.run("pip --version", shell=True, 
                            capture_output=True).returncode == 0
    has_conda = subprocess.run("conda --version", shell=True, 
                              capture_output=True).returncode == 0
    
    print(f"   {'✅' if has_pip else '❌'} pip disponible")
    print(f"   {'✅' if has_conda else '❌'} conda disponible")
    
    return has_cuda, has_pip, has_conda

def install_pytorch(has_cuda, has_pip, has_conda):
    """Installer PyTorch selon configuration"""
    print("\n🔥 INSTALLATION PYTORCH")
    
    if has_conda:
        print("   📦 Utilisation conda (recommandé)")
        if has_cuda:
            cmd = "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
        else:
            cmd = "conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
    else:
        print("   📦 Utilisation pip")
        if has_cuda:
            cmd = "pip install torch torchvision torchaudio"
        else:
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(cmd, "Installation PyTorch")

def install_other_deps(has_pip, has_conda):
    """Installer autres dépendances"""
    print("\n📦 INSTALLATION AUTRES DÉPENDANCES")
    
    deps = [
        "mujoco",
        "numpy", 
        "scipy",
        "matplotlib",
        "PyYAML",
        "tensorboard"
    ]
    
    success = True
    
    for dep in deps:
        if has_conda:
            # Essayer conda d'abord
            cmd = f"conda install -c conda-forge {dep} -y"
            if not run_command(cmd, f"Installation {dep} (conda)"):
                # Fallback pip
                cmd = f"pip install {dep}"
                if not run_command(cmd, f"Installation {dep} (pip fallback)"):
                    success = False
        else:
            cmd = f"pip install {dep}"
            if not run_command(cmd, f"Installation {dep}"):
                success = False
    
    return success

def verify_installation():
    """Vérifier installation"""
    print("\n🔍 VÉRIFICATION INSTALLATION")
    
    # Test imports critiques
    imports_to_test = [
        ("torch", "PyTorch"),
        ("mujoco", "MuJoCo"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML")
    ]
    
    success = True
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"   ✅ {name}: OK")
        except ImportError:
            print(f"   ❌ {name}: ÉCHEC")
            success = False
    
    # Test MuJoCo avec modèle
    if success:
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_path("results/g1_combined.xml")
            print("   ✅ Test modèle G1: OK")
        except Exception as e:
            print(f"   ❌ Test modèle G1: {e}")
            success = False
    
    return success

def main():
    """Installation automatique complète"""
    print("📦 INSTALLATION AUTOMATIQUE SACPER ULTRA")
    print("="*50)
    
    # Détection système
    has_cuda, has_pip, has_conda = detect_system()
    
    if not (has_pip or has_conda):
        print("❌ ERREUR: Ni pip ni conda disponible")
        return 1
    
    # Installation PyTorch
    if not install_pytorch(has_cuda, has_pip, has_conda):
        print("❌ ERREUR: Installation PyTorch échouée")
        return 1
    
    # Installation autres dépendances
    if not install_other_deps(has_pip, has_conda):
        print("❌ ERREUR: Installation dépendances échouée")
        return 1
    
    # Vérification
    if verify_installation():
        print("\n🎉 INSTALLATION RÉUSSIE!")
        print("\n🚀 PROCHAINES ÉTAPES:")
        print("   1. Vérifier système: python3 check_requirements.py")
        print("   2. Lancer entraînement: python3 launch_training.py --config cpu")
        return 0
    else:
        print("\n❌ VÉRIFICATION ÉCHOUÉE")
        print("\n💡 INSTALLATION MANUELLE:")
        print("   pip install torch torchvision torchaudio mujoco numpy scipy PyYAML")
        return 1

if __name__ == "__main__":
    exit(main())
