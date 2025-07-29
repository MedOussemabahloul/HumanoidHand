#!/usr/bin/env python3
"""
Script de test de stabilité pour G1 Fingers Optimisé
Reproduit l'erreur frictionloss exacte mentionnée par l'utilisateur
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_stability():
    """Test de stabilité du modèle G1 combiné"""
    
    print("=" * 60)
    print("🚀 TESTS DE STABILITÉ G1 FINGERS OPTIMISÉ")
    print("=" * 60)
    
    model_path = "results/g1_combined.xml"
    
    try:
        import mujoco
        
        print(f"🔍 Test de chargement: {model_path}")
        
        # Tentative de chargement du modèle
        model = mujoco.MjModel.from_xml_path(model_path)
        
        print("✅ Modèle chargé avec succès!")
        print(f"   - Nombre de corps: {model.nbody}")
        print(f"   - Nombre de joints: {model.njnt}")
        print(f"   - Nombre d'actuateurs: {model.nu}")
        
        # Créer les données de simulation
        data = mujoco.MjData(model)
        
        # Test de simulation courte
        print("🔄 Test de simulation (100 steps)...")
        for i in range(100):
            mujoco.mj_step(model, data)
        
        print("✅ Simulation réussie!")
        
        return True
        
    except ImportError:
        print("❌ MuJoCo n'est pas installé")
        print("   Installation: pip install mujoco")
        return False
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return False

if __name__ == "__main__":
    success = test_stability()
    sys.exit(0 if success else 1)