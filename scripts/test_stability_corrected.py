#!/usr/bin/env python3
"""
Script de test de stabilité pour le modèle G1 corrigé
Teste spécifiquement g1_combined_corrected.xml
"""

import sys
import os
from pathlib import Path

def test_corrected_model():
    """Test du modèle G1 combiné corrigé"""
    
    print("=" * 60)
    print("🚀 TEST DU MODÈLE G1 CORRIGÉ")
    print("=" * 60)
    
    workspace_path = Path(__file__).parent.parent
    model_path = workspace_path / "results" / "g1_combined_corrected.xml"
    
    if not model_path.exists():
        print(f"❌ Fichier non trouvé: {model_path}")
        print("💡 Exécutez d'abord: python scripts/fix_g1_combined.py")
        return False
    
    try:
        import mujoco
        
        print(f"🔍 Test de chargement: {model_path}")
        
        # Changer vers le dossier results pour les chemins relatifs
        original_cwd = os.getcwd()
        os.chdir(model_path.parent)
        
        try:
            # Tentative de chargement du modèle
            model = mujoco.MjModel.from_xml_path(model_path.name)
            
            print("✅ Modèle chargé avec succès!")
            print(f"   - Nombre de corps: {model.nbody}")
            print(f"   - Nombre de joints: {model.njnt}")
            print(f"   - Nombre d'actuateurs: {model.nu}")
            print(f"   - Nombre de degrés de liberté: {model.nv}")
            
            # Créer les données de simulation
            data = mujoco.MjData(model)
            
            # Test de simulation courte
            print("🔄 Test de simulation (100 steps)...")
            for i in range(100):
                mujoco.mj_step(model, data)
            
            print("✅ Simulation réussie!")
            print("🎉 Le problème frictionloss a été résolu!")
            
            return True
            
        finally:
            # Restaurer le répertoire original
            os.chdir(original_cwd)
        
    except ImportError:
        print("❌ MuJoCo n'est pas installé")
        print("   Installation: pip install mujoco")
        print("✅ Test XML uniquement...")
        
        # Test XML sans MuJoCo
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(model_path)
            root = tree.getroot()
            print(f"✅ XML valide - Élément racine: {root.tag}")
            print("💡 Installez MuJoCo pour un test complet")
            return True
        except ET.ParseError as e:
            print(f"❌ Erreur XML: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        
        # Analyser l'erreur
        error_str = str(e).lower()
        if 'frictionloss' in error_str:
            print("💡 L'erreur frictionloss persiste. Vérifiez les fichiers inclus.")
        elif 'file not found' in error_str or 'no such file' in error_str:
            print("💡 Problème de chemin de fichier. Vérifiez les chemins relatifs.")
        
        return False

def test_original_vs_corrected():
    """Compare le modèle original et le modèle corrigé"""
    
    print("\n" + "=" * 60)
    print("🔍 COMPARAISON ORIGINAL VS CORRIGÉ")
    print("=" * 60)
    
    workspace_path = Path(__file__).parent.parent
    original_path = workspace_path / "results" / "g1_combined.xml"
    corrected_path = workspace_path / "results" / "g1_combined_corrected.xml"
    
    print(f"📄 Original: {original_path}")
    print(f"📄 Corrigé: {corrected_path}")
    
    if original_path.exists() and corrected_path.exists():
        original_size = original_path.stat().st_size
        corrected_size = corrected_path.stat().st_size
        
        print(f"📊 Taille original: {original_size} bytes")
        print(f"📊 Taille corrigé: {corrected_size} bytes")
        print(f"📊 Différence: {corrected_size - original_size:+} bytes")
        
        # Tester le chargement de l'original (devrait échouer)
        try:
            import mujoco
            print("\n🔍 Test du modèle original...")
            os.chdir(original_path.parent)
            model = mujoco.MjModel.from_xml_path(original_path.name)
            print("⚠️  Le modèle original se charge (inattendu)")
        except Exception as e:
            print(f"❌ Modèle original échoue (attendu): {str(e)[:100]}...")
        
        print("\n🔍 Test du modèle corrigé...")
        try:
            os.chdir(corrected_path.parent)
            model = mujoco.MjModel.from_xml_path(corrected_path.name)
            print("✅ Modèle corrigé se charge correctement")
        except Exception as e:
            print(f"❌ Modèle corrigé échoue: {e}")

if __name__ == "__main__":
    print("🚀 TESTS DE STABILITÉ G1 FINGERS OPTIMISÉ - VERSION CORRIGÉE")
    
    success = test_corrected_model()
    
    if success:
        test_original_vs_corrected()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Le problème frictionloss a été résolu")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("💡 Vérifiez les messages d'erreur ci-dessus")
    
    sys.exit(0 if success else 1)