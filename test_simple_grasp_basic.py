#!/usr/bin/env python3
"""
Test basique du système de saisie G1 - Version locale
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Vérifie la structure des dossiers"""
    print("📁 Test de la structure...")
    
    required_dirs = ["envs", "agents", "utils", "results", "training_results"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/: OK")
        else:
            print(f"❌ {dir_name}/: Manquant")
    
    return True

def test_files():
    """Vérifie que les fichiers principaux existent"""
    print("\n📄 Test des fichiers...")
    
    required_files = [
        "envs/simple_grasp_env.py",
        "agents/improved_sac_agent.py",
        "utils/video_recorder.py", 
        "train_simple_grasp.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: OK")
        else:
            print(f"❌ {file_path}: Manquant")
            all_good = False
    
    return all_good

def test_model():
    """Vérifie la présence du modèle"""
    print("\n🤖 Test du modèle...")
    
    model_path = Path("results/g1_combined.xml")
    if model_path.exists():
        print(f"✅ Modèle trouvé: {model_path}")
        return True
    else:
        print(f"❌ Modèle manquant: {model_path}")
        print("💡 Placez votre modèle g1_combined.xml dans le dossier results/")
        return False

def test_dependencies():
    """Teste les dépendances Python"""
    print("\n📦 Test des dépendances...")
    
    deps = {
        "numpy": "numpy",
        "torch": "torch", 
        "gymnasium": "gymnasium",
        "mujoco": "mujoco"
    }
    
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError:
            print(f"❌ {name}: Manquant")
            missing.append(name)
    
    if missing:
        print(f"\n💡 Pour installer les dépendances manquantes:")
        print(f"   pip install {' '.join(missing)}")
    
    return len(missing) == 0

def create_sample_config():
    """Crée un exemple de configuration"""
    print("\n⚙️  Création de la configuration...")
    
    config = {
        "model_path": "results/g1_combined.xml",
        "episodes": 100,
        "learning_rate": 0.0003,
        "output_dir": "training_results"
    }
    
    try:
        import json
        with open("config_example.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ config_example.json créé")
    except Exception as e:
        print(f"⚠️  Erreur lors de la création: {e}")

def main():
    """Test principal"""
    print("🤖 TEST DU SYSTÈME DE SAISIE G1")
    print("=" * 50)
    
    # Tests
    structure_ok = test_directory_structure()
    files_ok = test_files()
    model_ok = test_model()  
    deps_ok = test_dependencies()
    
    # Créer un exemple de config
    create_sample_config()
    
    print("\n" + "=" * 50)
    
    if files_ok and structure_ok:
        print("✅ SYSTÈME PRÊT!")
        print("\n🚀 Pour lancer l'entraînement:")
        print("   python3 train_simple_grasp.py --episodes 100")
        
        if not model_ok:
            print("\n⚠️  N'oubliez pas de placer votre modèle dans results/")
        
        if not deps_ok:
            print("\n⚠️  Installez d'abord les dépendances manquantes")
            
    else:
        print("❌ CONFIGURATION INCOMPLÈTE")
        print("   Relancez setup_local_training.py")
    
    return files_ok and structure_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
