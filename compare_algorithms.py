#!/usr/bin/env python3
"""
🏆 COMPARAISON TD3 vs SAC vs PPO
==============================

Script pour comparer les trois algorithmes sur la même tâche de grasping.
"""

import time
import numpy as np
from pathlib import Path
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from envs.simple_robust_grasp_env import SimpleRobustGraspEnv

def test_algorithm(algo_name, model_class, env, timesteps=10000):
    """Test un algorithme spécifique"""
    print(f"\n🧪 Test de {algo_name}")
    print("-" * 40)
    
    # Configuration selon l'algorithme
    if algo_name == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]), 
            sigma=0.1 * np.ones(env.action_space.shape[-1])
        )
        model = model_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=50000,  # Réduit pour test rapide
            batch_size=128,
            tau=0.02,
            gamma=0.98,
            action_noise=action_noise,
            verbose=0
        )
    
    elif algo_name == "SAC":
        model = model_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=128,
            tau=0.02,
            gamma=0.98,
            ent_coef='auto',  # Exploration automatique
            verbose=0
        )
    
    elif algo_name == "PPO":
        model = model_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,    # Nombre de steps par batch
            batch_size=64,
            n_epochs=10,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0
        )
    
    # Entraînement
    start_time = time.time()
    try:
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        
        # Test de performance
        test_rewards = []
        obs, _ = env.reset()
        
        for _ in range(100):  # 100 steps de test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            test_rewards.append(reward)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        mean_reward = np.mean(test_rewards)
        
        print(f"✅ {algo_name} - Temps: {training_time:.1f}s, Reward moyen: {mean_reward:.2f}")
        
        # Sauvegarder le modèle
        results_dir = Path("algorithm_comparison")
        results_dir.mkdir(exist_ok=True)
        model.save(results_dir / f"{algo_name.lower()}_model")
        
        return {
            'algorithm': algo_name,
            'training_time': training_time,
            'mean_reward': mean_reward,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ {algo_name} - Erreur: {e}")
        return {
            'algorithm': algo_name,
            'training_time': 0,
            'mean_reward': -999,
            'success': False,
            'error': str(e)
        }

def main():
    print("🏆 COMPARAISON DES ALGORITHMES DE RL")
    print("=" * 50)
    print("Test rapide de TD3, SAC et PPO sur le grasping")
    print()
    
    # Créer l'environnement
    env = SimpleRobustGraspEnv(eval_mode=False)
    env = Monitor(env)
    
    # Algorithmes à tester
    algorithms = [
        ("TD3", TD3),
        ("SAC", SAC), 
        ("PPO", PPO)
    ]
    
    results = []
    timesteps = 5000  # Test rapide
    
    print(f"🔧 Configuration de test: {timesteps} timesteps par algorithme")
    
    # Tester chaque algorithme
    for algo_name, algo_class in algorithms:
        try:
            # Nouvel environnement pour chaque test
            test_env = SimpleRobustGraspEnv(eval_mode=False)
            test_env = Monitor(test_env)
            
            result = test_algorithm(algo_name, algo_class, test_env, timesteps)
            results.append(result)
            
            test_env.close()
            
        except Exception as e:
            print(f"❌ Erreur avec {algo_name}: {e}")
            results.append({
                'algorithm': algo_name,
                'training_time': 0,
                'mean_reward': -999,
                'success': False,
                'error': str(e)
            })
    
    env.close()
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS DE LA COMPARAISON")
    print("="*60)
    
    # Trier par performance
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        successful_results.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        print("\n🏆 CLASSEMENT (par reward moyen):")
        for i, result in enumerate(successful_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{medal} {i}. {result['algorithm']:>3} - "
                  f"Reward: {result['mean_reward']:>6.2f} - "
                  f"Temps: {result['training_time']:>5.1f}s")
        
        # Recommandation
        best = successful_results[0]
        print(f"\n💡 RECOMMANDATION: {best['algorithm']}")
        print(f"   Meilleure performance avec {best['mean_reward']:.2f} de reward moyen")
        
        # Analyse
        print(f"\n📈 ANALYSE:")
        print(f"   Le plus rapide: {min(successful_results, key=lambda x: x['training_time'])['algorithm']}")
        print(f"   Le plus performant: {best['algorithm']}")
        
        if any(r['algorithm'] == 'TD3' for r in successful_results):
            td3_result = next(r for r in successful_results if r['algorithm'] == 'TD3')
            print(f"   TD3 (actuel): {td3_result['mean_reward']:.2f} reward")
    
    if failed_results:
        print(f"\n❌ ÉCHECS:")
        for result in failed_results:
            print(f"   {result['algorithm']}: {result.get('error', 'Erreur inconnue')}")
    
    print(f"\n💾 Modèles sauvegardés dans: algorithm_comparison/")
    print(f"🔧 Pour un test plus long, modifiez timesteps dans le script")

if __name__ == "__main__":
    main()