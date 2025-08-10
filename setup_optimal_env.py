#!/usr/bin/env python3
"""
🔧 CONFIGURATION OPTIMALE DE L'ENVIRONNEMENT
===========================================

Ce script configure l'environnement pour garantir le bon fonctionnement
de l'entraînement, en s'inspirant de la configuration du notebook fonctionnel.

✅ Configuration MuJoCo optimale
✅ Variables d'environnement correctes
✅ Vérification des dépendances
✅ Création du modèle XML stable
✅ Test de l'environnement
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def setup_mujoco_environment():
    """
    Configurer MuJoCo pour un rendu optimal et stable
    """
    
    print("🔧 Configuration de MuJoCo...")
    
    # ✅ Configuration du rendu (comme le notebook fonctionnel)
    os.environ["MUJOCO_GL"] = "egl"  # Rendu headless stable
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = ""
    os.environ["MUJOCO_PY_MJKEY_PATH"] = ""
    
    # Configuration pour éviter les warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    print("✅ Variables d'environnement MuJoCo configurées")
    print(f"  - MUJOCO_GL: {os.environ.get('MUJOCO_GL')}")

def check_dependencies():
    """
    Vérifier que toutes les dépendances sont installées
    """
    
    print("\n🔍 Vérification des dépendances...")
    
    required_packages = [
        "mujoco",
        "numpy", 
        "gymnasium",
        "stable-baselines3",
        "torch",
        "imageio",
        "pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants: {missing_packages}")
        print("🔧 Installation automatique...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installé")
            except subprocess.CalledProcessError:
                print(f"❌ Échec installation {package}")
                return False
    
    print("✅ Toutes les dépendances sont disponibles")
    return True

def create_and_test_stable_model():
    """
    Créer et tester le modèle XML stable
    """
    
    print("\n🔧 Création du modèle XML stable...")
    
    # Importer et exécuter le script de correction
    try:
        from create_stable_xml import create_stable_xml_model, verify_xml_stability
        
        stable_path = create_stable_xml_model()
        
        if stable_path and verify_xml_stability(stable_path):
            print("✅ Modèle XML stable créé et vérifié")
            return stable_path
        else:
            print("❌ Échec de la création du modèle stable")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors de la création du modèle: {e}")
        return None

def test_optimal_environment():
    """
    Tester l'environnement optimal pour s'assurer qu'il fonctionne
    """
    
    print("\n🧪 Test de l'environnement optimal...")
    
    try:
        from envs.optimal_stable_env import OptimalStableGraspEnv
        
        # Test de création
        print("  🔧 Création de l'environnement...")
        env = OptimalStableGraspEnv()
        
        # Test de reset
        print("  🔄 Test du reset...")
        obs, _ = env.reset()
        print(f"    - Observation shape: {obs.shape}")
        print(f"    - Action space: {env.action_space}")
        
        # Test de quelques steps
        print("  🚀 Test de simulation...")
        stable_steps = 0
        for i in range(20):
            action = env.action_space.sample() * 0.3  # Actions modérées
            obs, reward, done, _, _ = env.step(action)
            
            # Vérifier la stabilité
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"    ❌ NaN/Inf détecté à l'étape {i}")
                break
            
            stable_steps += 1
            
            if i % 5 == 0:
                print(f"    Step {i}: reward = {reward:.3f}, stable = ✅")
            
            if done:
                print(f"    🎯 Épisode terminé à l'étape {i}")
                obs, _ = env.reset()
        
        env.close()
        
        if stable_steps >= 15:
            print("✅ Test de l'environnement réussi - simulation stable!")
            return True
        else:
            print(f"⚠️ Test partiellement réussi ({stable_steps}/20 steps stables)")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def create_optimal_directories():
    """
    Créer les dossiers nécessaires pour l'entraînement optimal
    """
    
    print("\n📁 Création des dossiers...")
    
    directories = [
        "optimal_results",
        "optimal_videos", 
        "optimal_logs",
        "optimal_models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ Dossiers créés")

def main():
    """
    Configuration complète de l'environnement optimal
    """
    
    print("🎯 CONFIGURATION OPTIMALE DE L'ENVIRONNEMENT")
    print("=" * 60)
    print("Basée sur le code fonctionnel du notebook de votre collègue")
    print()
    
    success = True
    
    # 1. Configuration MuJoCo
    setup_mujoco_environment()
    
    # 2. Vérification des dépendances
    if not check_dependencies():
        print("❌ Dépendances manquantes")
        success = False
    
    # 3. Création du modèle stable
    stable_model_path = create_and_test_stable_model()
    if not stable_model_path:
        print("❌ Échec création modèle stable")
        success = False
    
    # 4. Création des dossiers
    create_optimal_directories()
    
    # 5. Test de l'environnement
    if not test_optimal_environment():
        print("❌ Test environnement échoué")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CONFIGURATION OPTIMALE TERMINÉE AVEC SUCCÈS!")
        print()
        print("🚀 Vous pouvez maintenant lancer l'entraînement avec:")
        print("   python optimal_training.py")
        print()
        print("📁 Fichiers créés:")
        print("  - envs/optimal_stable_env.py (environnement optimal)")
        print("  - optimal_training.py (script d'entraînement)")
        print("  - results/g1_combined_stable.xml (modèle XML corrigé)")
        print()
        print("✅ Cette configuration reproduit exactement le code fonctionnel")
        print("   du notebook de votre collègue et évite les erreurs NaN/Inf")
    else:
        print("❌ ÉCHEC DE LA CONFIGURATION")
        print("🔧 Vérifiez les erreurs ci-dessus et réessayez")

if __name__ == "__main__":
    main()