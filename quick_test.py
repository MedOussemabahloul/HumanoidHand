#!/usr/bin/env python3
"""
🔧 TEST RAPIDE DE L'ENVIRONNEMENT
================================

Script pour vérifier que l'environnement optimisé se charge correctement
sans les dépendances d'entraînement.
"""

import os
import sys

def test_imports():
    """Test des imports essentiels"""
    print("🔍 Test des imports...")
    
    try:
        import mujoco
        print("✅ MuJoCo importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur import MuJoCo: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur import NumPy: {e}")
        return False
    
    return True

def test_xml_files():
    """Test de la disponibilité des fichiers XML"""
    print("\n📁 Test des fichiers XML...")
    
    xml_files = [
        "results/g1_combined.xml",
        "results/g1_combined_fixed.xml", 
        "results/g1_combined_balanced.xml"
    ]
    
    found_files = []
    for xml_file in xml_files:
        if os.path.exists(xml_file):
            print(f"✅ Trouvé: {xml_file}")
            found_files.append(xml_file)
        else:
            print(f"❌ Manquant: {xml_file}")
    
    return found_files

def test_mujoco_load(xml_file):
    """Test de chargement MuJoCo"""
    print(f"\n🤖 Test chargement MuJoCo: {xml_file}")
    
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
        print(f"✅ Modèle chargé avec succès!")
        print(f"   - Actuators: {model.nu}")
        print(f"   - Bodies: {model.nbody}")
        print(f"   - Joints: {model.njnt}")
        return True
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return False

def main():
    """Test principal"""
    print("=" * 50)
    print("🔧 TEST RAPIDE ENVIRONNEMENT GRASPING")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Échec des imports. Activez l'environnement virtuel:")
        print("   source venv/bin/activate  # ou")
        print("   conda activate votre_env")
        return False
    
    # Test fichiers XML
    xml_files = test_xml_files()
    if not xml_files:
        print("\n❌ Aucun fichier XML trouvé dans results/")
        return False
    
    # Test chargement MuJoCo
    for xml_file in xml_files:
        if test_mujoco_load(xml_file):
            print(f"\n🎉 SUCCÈS! Le modèle {xml_file} fonctionne parfaitement!")
            print("\n🚀 Vous pouvez maintenant lancer:")
            print("   python optimized_train1.py")
            return True
    
    print("\n❌ Aucun modèle XML ne se charge correctement")
    print("💡 Vérifiez les chemins dans les fichiers XML inclus")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)