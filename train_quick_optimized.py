#!/usr/bin/env python3
"""
⚡ ENTRAÎNEUR SAC RAPIDE ET OPTIMISÉ
==================================
Version rapide (10,000 timesteps) pour tests rapides
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/workspace/envs')

from envs.curriculum_grasp_env import CurriculumGraspEnv
from stable_baselines3 import SAC

def main():
    print("⚡ ENTRAÎNEMENT SAC RAPIDE ET OPTIMISÉ")
    print("=" * 50)
    
    # Environnement
    print("🏗️  Création environnement...")
    env = CurriculumGraspEnv()
    env.current_level = 1
    
    # Modèle optimisé mais rapide
    print("🧠 Création modèle SAC optimisé...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=0.0001,    # Lent mais stable
        buffer_size=10000,       # Petit pour rapidité
        batch_size=64,           # Petit
        tau=0.01,
        gamma=0.98,
        train_freq=4,
        verbose=1
    )
    
    # Entraînement rapide
    print("📚 Entraînement (10,000 timesteps)...")
    model.learn(total_timesteps=10000, log_interval=4)
    
    # Test
    print("\n🎮 Test du modèle...")
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.3f}, total={total_reward:.2f}")
        
        if done or trunc:
            break
    
    print(f"\n📊 RÉSULTATS:")
    print(f"Récompense totale: {total_reward:.2f}")
    
    if total_reward > -30:
        print("✅ SUCCÈS! Le modèle progresse!")
    else:
        print("⚠️  Nécessite plus d'entraînement")
    
    env.close()
    print("✅ Terminé!")

if __name__ == "__main__":
    main()