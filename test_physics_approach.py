#!/usr/bin/env python3
"""
TEST DE L'APPROCHE PHYSIQUE vs BLOCAGE
Compare les deux approches pour résoudre l'instabilité des doigts
"""

import sys
from pathlib import Path
sys.path.append('.')
sys.path.append('./envs')

def test_physics_vs_blocking():
    """Compare les deux approches"""
    print("🔬 COMPARAISON APPROCHES: PHYSIQUE vs BLOCAGE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Approche PHYSIQUE (doigts actifs)
    print("\n🧪 TEST 1: APPROCHE PHYSIQUE (Doigts ACTIFS)")
    print("-" * 50)
    
    try:
        from envs.corrected_physics_env import CorrectedPhysicsGraspEnv
        
        env_physics = CorrectedPhysicsGraspEnv(
            xml_path="results/g1_combined.xml",
            max_episode_steps=10,
            fix_physics=True
        )
        
        print(f"\n📊 CONFIGURATION PHYSIQUE:")
        print(f"   🖐️  Doigts: {len(env_physics.finger_dofs)} DOFs (ACTIFS)")
        print(f"   💪 Bras: {len(env_physics.arm_dofs)} DOFs")
        print(f"   🎯 Contrôlables: {len(env_physics.controllable_dofs)} DOFs")
        print(f"   🎯 Actions: {env_physics.action_space.shape}")
        
        # Test de stabilité
        instabilities = 0
        successful_steps = 0
        
        for episode in range(3):
            obs, _ = env_physics.reset()
            print(f"\n   Episode {episode+1}/3:")
            
            for step in range(10):
                action = env_physics.action_space.sample() * 0.1  # Actions petites
                obs, reward, term, trunc, info = env_physics.step(action)
                
                if info.get('instability_count', 0) > 0:
                    instabilities += info['instability_count']
                    print(f"     Step {step+1}: ❌ Instabilité détectée")
                    break
                else:
                    successful_steps += 1
                    print(f"     Step {step+1}: ✅ Stable (reward: {reward:.1f})")
                
                if term or trunc:
                    break
        
        env_physics.close()
        
        results['physics'] = {
            'instabilities': instabilities,
            'successful_steps': successful_steps,
            'finger_dofs': len(env_physics.finger_dofs),
            'controllable_dofs': len(env_physics.controllable_dofs)
        }
        
        print(f"\n📈 RÉSULTATS APPROCHE PHYSIQUE:")
        print(f"   ⚠️  Instabilités: {instabilities}")
        print(f"   ✅ Steps réussis: {successful_steps}")
        print(f"   🖐️  Doigts actifs: {len(env_physics.finger_dofs)}")
        
    except Exception as e:
        print(f"❌ Erreur approche physique: {e}")
        results['physics'] = {'error': str(e)}
    
    # Test 2: Approche BLOCAGE (doigts bloqués)
    print("\n🧪 TEST 2: APPROCHE BLOCAGE (Doigts BLOQUÉS)")
    print("-" * 50)
    
    try:
        from envs.corrected_ultra_stable_env import CorrectedUltraStableGraspEnv
        
        env_blocking = CorrectedUltraStableGraspEnv(
            xml_path="results/g1_combined.xml",
            max_episode_steps=10,
            block_fingers=True
        )
        
        print(f"\n📊 CONFIGURATION BLOCAGE:")
        print(f"   🖐️  Doigts: {len(env_blocking.finger_dofs)} DOFs (BLOQUÉS)")
        print(f"   💪 Bras: {len(env_blocking.arm_dofs)} DOFs")
        print(f"   🎯 Contrôlables: {len(env_blocking.controllable_dofs)} DOFs")
        print(f"   🎯 Actions: {env_blocking.action_space.shape}")
        
        # Test de stabilité
        instabilities = 0
        successful_steps = 0
        
        for episode in range(3):
            obs, _ = env_blocking.reset()
            print(f"\n   Episode {episode+1}/3:")
            
            for step in range(10):
                action = env_blocking.action_space.sample() * 0.1
                obs, reward, term, trunc, info = env_blocking.step(action)
                
                if info.get('instability_count', 0) > 0:
                    instabilities += info['instability_count']
                    print(f"     Step {step+1}: ❌ Instabilité détectée")
                    break
                else:
                    successful_steps += 1
                    print(f"     Step {step+1}: ✅ Stable (reward: {reward:.1f})")
                
                if term or trunc:
                    break
        
        env_blocking.close()
        
        results['blocking'] = {
            'instabilities': instabilities,
            'successful_steps': successful_steps,
            'finger_dofs': len(env_blocking.finger_dofs),
            'controllable_dofs': len(env_blocking.controllable_dofs)
        }
        
        print(f"\n📈 RÉSULTATS APPROCHE BLOCAGE:")
        print(f"   ⚠️  Instabilités: {instabilities}")
        print(f"   ✅ Steps réussis: {successful_steps}")
        print(f"   🔒 Doigts bloqués: {len(env_blocking.finger_dofs)}")
        
    except Exception as e:
        print(f"❌ Erreur approche blocage: {e}")
        results['blocking'] = {'error': str(e)}
    
    # Comparaison
    print("\n" + "="*60)
    print("📊 COMPARAISON FINALE")
    print("="*60)
    
    if 'physics' in results and 'blocking' in results:
        if 'error' not in results['physics'] and 'error' not in results['blocking']:
            print(f"\n🧪 APPROCHE PHYSIQUE (Doigts ACTIFS):")
            print(f"   ⚠️  Instabilités: {results['physics']['instabilities']}")
            print(f"   ✅ Steps réussis: {results['physics']['successful_steps']}")
            print(f"   🎯 DOFs contrôlables: {results['physics']['controllable_dofs']}")
            
            print(f"\n🛡️  APPROCHE BLOCAGE (Doigts BLOQUÉS):")
            print(f"   ⚠️  Instabilités: {results['blocking']['instabilities']}")
            print(f"   ✅ Steps réussis: {results['blocking']['successful_steps']}")
            print(f"   🎯 DOFs contrôlables: {results['blocking']['controllable_dofs']}")
            
            # Recommandation
            physics_stable = results['physics']['instabilities'] == 0
            blocking_stable = results['blocking']['instabilities'] == 0
            
            print(f"\n🎯 RECOMMANDATION:")
            if physics_stable and blocking_stable:
                print("   ✅ Les DEUX approches fonctionnent!")
                print("   💡 Utilisez l'approche PHYSIQUE pour plus de fonctionnalités")
                print("   🔧 Utilisez l'approche BLOCAGE pour plus de simplicité")
            elif physics_stable:
                print("   🏆 APPROCHE PHYSIQUE recommandée")
                print("   ✅ Stable ET utilise tous les DOFs")
            elif blocking_stable:
                print("   🛡️  APPROCHE BLOCAGE recommandée")
                print("   ✅ Stable mais fonctionnalité réduite")
            else:
                print("   ⚠️  Les deux approches ont des problèmes")
                print("   🔧 Investigation supplémentaire nécessaire")
    
    return results

def main():
    """Test principal"""
    print("🔬 ANALYSE DES SOLUTIONS POUR LES DOIGTS PROBLÉMATIQUES")
    print("\n💡 POURQUOI BLOQUER N'EST PAS LA SEULE SOLUTION:")
    print("   1. 🔧 Correction des paramètres physiques")
    print("   2. ⚙️  Ajustement des gains d'actuateurs") 
    print("   3. 🕰️  Modification du timestep")
    print("   4. 🔄 Amélioration du solver")
    print("   5. 🎛️  Contrôle adaptatif")
    
    results = test_physics_vs_blocking()
    
    print(f"\n📚 AUTRES ALTERNATIVES POSSIBLES:")
    print(f"   • 🎯 Contrôle PID adaptatif pour doigts")
    print(f"   • 🔄 Solver différent (RK4 vs Euler)")
    print(f"   • ⚡ Timestep variable")
    print(f"   • 🎛️  Gains d'actuateurs dynamiques")
    print(f"   • 🛡️  Contraintes de sécurité")
    print(f"   • 🔧 Modèle URDF optimisé")
    
    return results

if __name__ == "__main__":
    main()