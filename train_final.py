#!/usr/bin/env python3
"""
Script d'entraînement final - Simple et robuste
"""

import sys
import os
import time
import json
import numpy as np

# Setup path et environnement
sys.path.append('/workspace')
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYTHONWARNINGS"] = "ignore"

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from envs.professional_grasp_env import make_professional_env

def train_robot(algorithm="TD3", timesteps=20_000, model_name="final_model"):
    """
    Entraînement principal du robot
    
    Args:
        algorithm: "TD3" ou "SAC"
        timesteps: Nombre de steps d'entraînement
        model_name: Nom du modèle à sauvegarder
    """
    
    print(f"🤖 ENTRAÎNEMENT {algorithm} - {timesteps:,} steps")
    print("=" * 50)
    
    # 1. Créer l'environnement
    print("📁 Création de l'environnement...")
    env = make_professional_env()
    
    # 2. Test initial
    print("🧪 Test initial...")
    obs, _ = env.reset()
    total_reward = 0
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
    
    avg_reward = total_reward / 20
    print(f"✅ Test initial - Reward moyen: {avg_reward:.2f}")
    
    if avg_reward < -100:
        print("❌ Environnement instable - arrêt")
        return None
    
    # 3. Créer le modèle
    print(f"🧠 Création du modèle {algorithm}...")
    
    if algorithm == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.25 * np.ones(env.action_space.shape[0])
        )
        
        model = TD3(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=500_000,
            gamma=0.99,
            tau=0.005,
            verbose=1
        )
    
    elif algorithm == "SAC":
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=500_000,
            gamma=0.99,
            tau=0.005,
            verbose=1
        )
    
    else:
        print(f"❌ Algorithme non supporté: {algorithm}")
        return None
    
    # 4. Entraînement avec monitoring manuel
    print("🎓 DÉMARRAGE ENTRAÎNEMENT...")
    start_time = time.time()
    
    # Entraînement par phases pour monitoring
    save_freq = 5_000
    n_phases = max(1, timesteps // save_freq)
    
    best_reward = -np.inf
    
    for phase in range(n_phases):
        phase_steps = min(save_freq, timesteps - phase * save_freq)
        
        print(f"\n🔄 Phase {phase+1}/{n_phases} - {phase_steps:,} steps")
        
        # Entraînement de cette phase
        model.learn(
            total_timesteps=phase_steps,
            reset_num_timesteps=False
        )
        
        # Test de performance
        print("📊 Test de performance...")
        test_env = make_professional_env(eval_mode=True)
        
        test_rewards = []
        for episode in range(5):
            obs, _ = test_env.reset()
            episode_reward = 0
            
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = test_env.step(action)
                episode_reward += reward
                if term or trunc:
                    break
            
            test_rewards.append(episode_reward)
        
        avg_test_reward = np.mean(test_rewards)
        
        if avg_test_reward > best_reward:
            best_reward = avg_test_reward
            print(f"🎯 Nouveau record: {avg_test_reward:.2f}")
        
        print(f"📈 Performance phase {phase+1}: {avg_test_reward:.2f} (best: {best_reward:.2f})")
        
        # Sauvegarde
        phase_path = f"/workspace/results/{model_name}_phase_{phase+1}"
        model.save(phase_path)
        
        # Stats
        elapsed = time.time() - start_time
        steps_per_sec = (phase+1) * phase_steps / elapsed
        
        stats = {
            "phase": phase + 1,
            "total_phases": n_phases,
            "steps_completed": (phase + 1) * phase_steps,
            "total_steps": timesteps,
            "elapsed_minutes": elapsed / 60,
            "steps_per_sec": steps_per_sec,
            "avg_test_reward": avg_test_reward,
            "best_reward": best_reward,
            "algorithm": algorithm
        }
        
        with open("/workspace/results/training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        test_env.close()
    
    # 5. Sauvegarde finale
    final_path = f"/workspace/results/{model_name}_final"
    model.save(final_path)
    
    # 6. Évaluation finale
    print("\n🎯 ÉVALUATION FINALE...")
    final_env = make_professional_env(eval_mode=True)
    
    final_rewards = []
    final_distances = []
    
    for episode in range(10):
        obs, _ = final_env.reset()
        episode_reward = 0
        min_distance = np.inf
        
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = final_env.step(action)
            episode_reward += reward
            
            # Calculer distance (approximative)
            cube_pos = obs[:3]  # Position cube dans observation
            distance = np.linalg.norm(cube_pos - np.array([0.5, 0.2, 1.0]))
            min_distance = min(min_distance, distance)
            
            if term or trunc:
                break
        
        final_rewards.append(episode_reward)
        final_distances.append(min_distance)
        
        print(f"Épisode {episode+1}: reward={episode_reward:.2f}, min_distance={min_distance:.3f}")
    
    # Résultats finaux
    final_avg_reward = np.mean(final_rewards)
    final_avg_distance = np.mean(final_distances)
    
    print(f"\n🏆 RÉSULTATS FINAUX:")
    print(f"   Reward moyen: {final_avg_reward:.2f}")
    print(f"   Distance moyenne: {final_avg_distance:.3f}")
    print(f"   Meilleur épisode: {max(final_rewards):.2f}")
    
    # Sauvegarder résultats
    results = {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "final_avg_reward": final_avg_reward,
        "final_avg_distance": final_avg_distance,
        "best_episode_reward": max(final_rewards),
        "all_rewards": final_rewards,
        "all_distances": final_distances,
        "training_time_minutes": (time.time() - start_time) / 60
    }
    
    with open("/workspace/results/final_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    final_env.close()
    env.close()
    
    print(f"✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"📁 Modèle final: {final_path}")
    
    return final_path

def main():
    """Point d'entrée principal"""
    
    print("🚀 SYSTÈME D'ENTRAÎNEMENT ROBOTIQUE FINAL")
    print("=" * 50)
    
    # Vérifier le modèle
    model_path = "/workspace/results/g1_combined_balanced.xml"
    if not os.path.exists(model_path):
        print("❌ Modèle manquant! Création...")
        os.system('python3 -c "import re; xml=open(\'/workspace/results/g1_combined.xml\').read(); xml=re.sub(r\'timestep=\\\"0\\.0005\\\"\', \'timestep=\\\"0.005\\\"\', xml); xml=re.sub(r\'solver=\\\"Newton\\\"\', \'solver=\\\"PGS\\\"\', xml); xml=re.sub(r\'iterations=\\\"500\\\"\', \'iterations=\\\"100\\\"\', xml); xml=re.sub(r\'tolerance=\\\"1e-12\\\"\', \'tolerance=\\\"1e-8\\\"\', xml); xml=re.sub(r\'kp=\\\"120\\\" kv=\\\"25\\\"\', \'kp=\\\"80\\\" kv=\\\"20\\\"\', xml); xml=re.sub(r\'kp=\\\"50\\\" kv=\\\"15\\\"\', \'kp=\\\"25\\\" kv=\\\"10\\\"\', xml); xml=re.sub(r\'forcerange=\\\"-150 150\\\"\', \'forcerange=\\\"-100 100\\\"\', xml); xml=re.sub(r\'forcerange=\\\"-50 50\\\"\', \'forcerange=\\\"-25 25\\\"\', xml); open(\'/workspace/results/g1_combined_balanced.xml\', \'w\').write(xml); print(\'✅ Modèle créé\')"')
    
    print("📋 DÉMARRAGE ENTRAÎNEMENT:")
    print("🎯 Algorithme: TD3")
    print("📊 Steps: 20,000")
    print("⏱️ Durée estimée: 10-15 minutes")
    print()
    
    try:
        model_path = train_robot(
            algorithm="TD3",
            timesteps=20_000,
            model_name="professional_robot"
        )
        
        if model_path:
            print("\n🎉 SUCCÈS COMPLET!")
            print(f"📁 Modèle: {model_path}")
            print("📊 Résultats: /workspace/results/final_evaluation.json")
            print("📊 Stats: /workspace/results/training_stats.json")
            
            print("\n🚀 POUR UTILISER LE MODÈLE:")
            print(f"model = TD3.load('{model_path}')")
            print("obs, _ = env.reset()")
            print("action, _ = model.predict(obs)")
            
        else:
            print("❌ Entraînement échoué")
            return 1
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())