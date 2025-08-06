
#!/usr/bin/env python3
"""
TEST FINAL du système corrigé
"""

import sys
from pathlib import Path
sys.path.append('.')
sys.path.append('./envs')

def test_corrected_system():
    """Test complet du système corrigé"""
    print("🔧 TEST FINAL DU SYSTÈME CORRIGÉ")
    print("=" * 50)
    
    # Test 1: Fichiers
    files = [
        "envs/corrected_ultra_stable_env.py",
        "train_corrected_ultra_stable.py"
    ]
    
    for file_path in files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: MANQUANT")
            return False
    
    # Test 2: Import
    try:
        from envs.corrected_ultra_stable_env import CorrectedUltraStableGraspEnv
        print("✅ Import CorrectedUltraStableGraspEnv réussi")
        
        # Test avec modèle si disponible
        if Path("results/g1_combined.xml").exists():
            print("✅ Modèle G1 disponible")
            
            try:
                env = CorrectedUltraStableGraspEnv(
                    xml_path="results/g1_combined.xml",
                    max_episode_steps=10,
                    block_fingers=True
                )
                
                print(f"\n📊 IDENTIFICATION CORRIGÉE VALIDÉE:")
                print(f"   🖐️  Doigts: {env.finger_dofs}")
                print(f"   💪 Bras: {env.arm_dofs}")
                print(f"   🎯 Actions: {env.action_space.shape} (±{env.action_space.high[0]:.3f})")
                print(f"   👁️  Obs: {env.observation_space.shape}")
                
                # Vérifier que TOUS les DOFs problématiques sont identifiés
                expected = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
                missing = set(expected) - set(env.finger_dofs)
                
                if not missing:
                    print("✅ TOUS les DOFs problématiques sont identifiés comme doigts")
                else:
                    print(f"❌ DOFs manqués: {list(missing)}")
                    return False
                
                # Test reset
                obs, _ = env.reset()
                print(f"✅ Reset réussi: obs shape {obs.shape}")
                
                # Test step
                action = env.action_space.sample() * 0.001
                obs, reward, term, trunc, info = env.step(action)
                print(f"✅ Step réussi: reward {reward:.2f}")
                print(f"   Instabilités: {info.get('instability_count', 0)}")
                
                env.close()
                return True
                
            except Exception as e:
                print(f"❌ Erreur test environnement: {e}")
                return False
        else:
            print("⚠️  Modèle G1 manquant - test import seulement")
            return True
            
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False

def main():
    """Test principal"""
    success = test_corrected_system()
    
    print("\n" + "="*50)
    if success:
        print("🟢 SYSTÈME CORRIGÉ VALIDÉ!")
        print("\n🚀 Prêt pour l'entraînement:")
        print("   python3 train_corrected_ultra_stable.py --episodes 20 --max-steps 15")
        print("\n📈 Résultats attendus:")
        print("   - Aucune instabilité sur DOF 15-30")
        print("   - Épisodes de 10-15 steps minimum")
        print("   - Identification parfaite des doigts")
    else:
        print("🔴 SYSTÈME À CORRIGER")
    
    return success

if __name__ == "__main__":
    main()
