#!/usr/bin/env python3
"""
🔧 TEST RAPIDE DES CORRECTIONS
==============================

Test des corrections apportées pour :
1. Dimension d'observation correcte (88)
2. Récompenses positives
3. Stabilité améliorée
"""

import sys
import numpy as np

# Ajouter les chemins
sys.path.append('/home/oussema/Documents/project/envs')
sys.path.append('/workspace/envs')

try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
    print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

def test_observation_dimension():
    """Test de la dimension d'observation"""
    print("\n🔍 Test dimension d'observation...")
    
    env = CurriculumGraspEnv()
    
    # Test du reset
    obs, info = env.reset()
    print(f"  📊 Dimension après reset: {obs.shape}")
    
    # Test de plusieurs steps
    for i in range(5):
        action = env.action_space.sample() * 0.01
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: obs={obs.shape}, reward={reward:.3f}")
    
    env.close()
    
    if obs.shape[0] == 88:
        print("  ✅ Dimension d'observation correcte")
        return True
    else:
        print(f"  ❌ Dimension incorrecte: {obs.shape[0]} au lieu de 88")
        return False

def test_rewards():
    """Test du système de récompenses"""
    print("\n🎯 Test système de récompenses...")
    
    env = CurriculumGraspEnv()
    
    obs, info = env.reset()
    total_reward = 0
    positive_rewards = 0
    negative_rewards = 0
    
    for i in range(50):
        action = env.action_space.sample() * 0.02  # Actions très douces
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if reward > 0:
            positive_rewards += 1
        elif reward < 0:
            negative_rewards += 1
        
        if i % 10 == 0:
            print(f"  Step {i}: reward={reward:.3f}, stabilité={info['stability_count']}")
        
        if terminated or truncated:
            print(f"  🏁 Épisode terminé au step {i}")
            break
    
    env.close()
    
    print(f"  📊 Récompense totale: {total_reward:.2f}")
    print(f"  📈 Récompenses positives: {positive_rewards}/50")
    print(f"  📉 Récompenses négatives: {negative_rewards}/50")
    
    if total_reward > -10 and positive_rewards > 30:
        print("  ✅ Système de récompenses amélioré")
        return True
    else:
        print("  ⚠️ Système de récompenses à optimiser")
        return False

def test_stability():
    """Test de la stabilité générale"""
    print("\n⚖️ Test stabilité générale...")
    
    env = CurriculumGraspEnv()
    
    crashes = 0
    successful_episodes = 0
    
    for episode in range(3):  # Test 3 épisodes courts
        try:
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(100):
                action = env.action_space.sample() * 0.05
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"  Épisode {episode+1}: {step+1} steps, reward={episode_reward:.2f}")
            successful_episodes += 1
            
        except Exception as e:
            print(f"  ❌ Crash épisode {episode+1}: {e}")
            crashes += 1
    
    env.close()
    
    print(f"  📊 Épisodes réussis: {successful_episodes}/3")
    print(f"  💥 Crashes: {crashes}/3")
    
    if crashes == 0:
        print("  ✅ Stabilité excellente")
        return True
    else:
        print("  ⚠️ Stabilité à améliorer")
        return False

def test_sac_compatibility():
    """Test de compatibilité avec SAC"""
    print("\n🧠 Test compatibilité SAC...")
    
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env
        
        # Créer un environnement vectorisé pour SAC
        def make_env():
            return CurriculumGraspEnv()
        
        vec_env = make_vec_env(make_env, n_envs=1)
        
        # Créer un modèle SAC minimal
        model = SAC("MlpPolicy", vec_env, verbose=0, learning_starts=10)
        
        # Test d'une observation
        obs = vec_env.reset()
        print(f"  📊 Observation vectorisée: {obs.shape}")
        
        # Test de prédiction
        action, _ = model.predict(obs, deterministic=True)
        print(f"  🎯 Action prédite: {action.shape}")
        
        # Test d'un step
        obs, reward, done, info = vec_env.step(action)
        print(f"  🔄 Step réussi: obs={obs.shape}, reward={reward}")
        
        vec_env.close()
        
        print("  ✅ Compatibilité SAC parfaite")
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur compatibilité SAC: {e}")
        return False

def main():
    """Test principal"""
    print("🔧 TESTS DES CORRECTIONS APPLIQUÉES")
    print("=" * 50)
    
    tests = [
        ("Dimension d'observation", test_observation_dimension),
        ("Système de récompenses", test_rewards),
        ("Stabilité générale", test_stability),
        ("Compatibilité SAC", test_sac_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"  ❌ Erreur durant le test: {e}")
            results.append(False)
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ RÉUSSI" if results[i] else "❌ ÉCHOUÉ"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Taux de réussite: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed >= 3:
        print("🎉 CORRECTIONS RÉUSSIES! Prêt pour l'entraînement.")
        return True
    else:
        print("⚠️ Corrections partielles. Quelques problèmes subsistent.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)