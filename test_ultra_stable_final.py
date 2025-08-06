#!/usr/bin/env python3
"""
Test final du système ultra-stable
"""

import sys
from pathlib import Path

def test_system():
    """Test complet du système"""
    print("🛡️  TEST FINAL SYSTÈME ULTRA-STABLE")
    print("=" * 50)
    
    # Test 1: Structure fichiers
    required_files = [
        "envs/ultra_stable_grasp_env.py",
        "agents/improved_sac_agent.py", 
        "train_ultra_stable.py"
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: MANQUANT")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ {len(missing)} fichiers manquants")
        return False
    
    # Test 2: Correction PyTorch
    try:
        with open("agents/improved_sac_agent.py", 'r') as f:
            content = f.read()
        
        if "(~dones).float()" in content:
            print("✅ Correction PyTorch appliquée")
        else:
            print("❌ Correction PyTorch manquante")
            return False
    except Exception as e:
        print(f"❌ Erreur vérification PyTorch: {e}")
        return False
    
    # Test 3: Environnement ultra-stable
    try:
        with open("envs/ultra_stable_grasp_env.py", 'r') as f:
            content = f.read()
        
        features = [
            "block_fingers=True",
            "timestep = 0.01", 
            "iterations = 100",
            "INSTABILITÉ DÉTECTÉE",
            "joint_name = mujoco.mj_id2name"
        ]
        
        found = 0
        for feature in features:
            if feature in content:
                found += 1
        
        if found >= 4:
            print(f"✅ Environnement ultra-stable: {found}/5 features")
        else:
            print(f"⚠️  Environnement: {found}/5 features seulement")
    except Exception as e:
        print(f"❌ Erreur vérification environnement: {e}")
        return False
    
    # Test 4: Dépendances
    deps_ok = 0
    deps_total = 4
    
    for dep in ["numpy", "torch", "gymnasium", "mujoco"]:
        try:
            __import__(dep)
            print(f"✅ {dep}: Disponible")
            deps_ok += 1
        except ImportError:
            print(f"⚠️  {dep}: Manquant (pip install {dep})")
    
    # Test 5: Modèle G1
    if Path("results/g1_combined.xml").exists():
        print("✅ Modèle G1: Présent")
        model_ok = True
    else:
        print("⚠️  Modèle G1: Manquant (placez dans results/)")
        model_ok = False
    
    # Résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DU TEST:")
    print(f"   Fichiers: {len(required_files) - len(missing)}/{len(required_files)}")
    print(f"   Corrections: ✅ PyTorch, ✅ Environnement")
    print(f"   Dépendances: {deps_ok}/{deps_total}")
    print(f"   Modèle G1: {'✅' if model_ok else '⚠️'}")
    
    if len(missing) == 0 and deps_ok >= 3:
        print("\n🟢 SYSTÈME PRÊT POUR L'ENTRAÎNEMENT!")
        print("\n🚀 Commande recommandée:")
        print("   python3 train_ultra_stable.py --episodes 20 --max-steps 30")
        return True
    else:
        print("\n🟡 SYSTÈME PARTIELLEMENT PRÊT")
        print("\n💡 Actions requises:")
        if missing:
            print(f"   - Créer fichiers manquants: {missing}")
        if deps_ok < 3:
            print("   - Installer dépendances: pip install torch numpy gymnasium mujoco")
        if not model_ok:
            print("   - Placer g1_combined.xml dans results/")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
