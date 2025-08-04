#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_simple_grasp.py

Script de démonstration du système de grasping simplifié.
Montre le robot en train d'apprendre à saisir un cube.
"""

import os
import sys
import numpy as np
import time
import argparse

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.simple_grasp_env import SimpleGraspEnv
from agents.simple_sac_agent import SimpleSACAgent

def demo_random_agent():
    """
    Démonstration avec un agent aléatoire
    """
    print("🤖 DÉMONSTRATION - AGENT ALÉATOIRE")
    print("="*50)
    
    # Créer l'environnement
    env = SimpleGraspEnv(config={
        "touch_sensors": ["touch1_sensor", "touch2_sensor"],
        "cube_body_name": "cube",
        "max_steps_per_episode": 200
    })
    
    print(f"✅ Environnement créé")
    print(f"   - Observations: {env.observation_space.shape}")
    print(f"   - Actions: {env.action_space.shape}")
    print(f"   - Capteurs tactiles: {len(env.touch_ids)}")
    
    # Épisode de démonstration
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"\n🎯 Début de la démonstration...")
    print(f"   Position initiale du cube: {env.data.xpos[env.cube_id][:3]}")
    
    for step in range(200):
        # Action aléatoire
        action = env.action_space.sample()
        
        # Exécuter l'action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Afficher les informations périodiquement
        if step % 20 == 0:
            cube_height = info['cube_height']
            touch_values = info['touch_values']
            print(f"   Step {step:3d}: Reward={reward:6.2f}, Cube height={cube_height:.3f}, Touch={touch_values}")
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 Démonstration terminée")
    print(f"   Steps totaux: {step + 1}")
    print(f"   Récompense totale: {total_reward:.2f}")
    print(f"   Hauteur finale du cube: {info['cube_height']:.3f}")
    print(f"   Contact détecté: {any(v > 0.1 for v in info['touch_values'])}")
    
    env.close()

def demo_trained_agent(model_path):
    """
    Démonstration avec un agent entraîné
    """
    print("🤖 DÉMONSTRATION - AGENT ENTRÂINÉ")
    print("="*50)
    
    # Créer l'environnement
    env = SimpleGraspEnv(config={
        "touch_sensors": ["touch1_sensor", "touch2_sensor"],
        "cube_body_name": "cube",
        "max_steps_per_episode": 200
    })
    
    # Créer l'agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SimpleSACAgent(obs_dim, act_dim, device='cpu')
    
    # Charger le modèle entraîné
    try:
        agent.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # Épisode de démonstration
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"\n🎯 Début de la démonstration avec l'agent entraîné...")
    print(f"   Position initiale du cube: {env.data.xpos[env.cube_id][:3]}")
    
    for step in range(200):
        # Action de l'agent entraîné
        action = agent.select_action(obs, evaluate=True)
        
        # Exécuter l'action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Afficher les informations périodiquement
        if step % 20 == 0:
            cube_height = info['cube_height']
            touch_values = info['touch_values']
            print(f"   Step {step:3d}: Reward={reward:6.2f}, Cube height={cube_height:.3f}, Touch={touch_values}")
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 Démonstration terminée")
    print(f"   Steps totaux: {step + 1}")
    print(f"   Récompense totale: {total_reward:.2f}")
    print(f"   Hauteur finale du cube: {info['cube_height']:.3f}")
    print(f"   Contact détecté: {any(v > 0.1 for v in info['touch_values'])}")
    
    env.close()

def demo_training_progress():
    """
    Démonstration de la progression de l'entraînement
    """
    print("📈 DÉMONSTRATION - PROGRESSION DE L'ENTRAÎNEMENT")
    print("="*50)
    
    # Créer l'environnement
    env = SimpleGraspEnv(config={
        "touch_sensors": ["touch1_sensor", "touch2_sensor"],
        "cube_body_name": "cube",
        "max_steps_per_episode": 100
    })
    
    # Créer l'agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SimpleSACAgent(obs_dim, act_dim, hidden_sizes=[64, 64], device='cpu')
    
    print(f"✅ Agent créé pour l'entraînement")
    print(f"   - Observations: {obs_dim}")
    print(f"   - Actions: {act_dim}")
    
    # Entraînement rapide
    print(f"\n🎯 Début de l'entraînement rapide (5 épisodes)...")
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            # Action de l'agent (avec exploration)
            action = agent.select_action(obs, evaluate=False)
            
            # Exécuter l'action
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Stocker l'expérience
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Mettre à jour l'agent
        if len(agent.replay_buffer) >= 50:
            agent.update(batch_size=32)
        
        # Afficher les statistiques
        cube_height = info['cube_height']
        touch_values = info['touch_values']
        contact_detected = any(v > 0.1 for v in touch_values)
        
        print(f"   Episode {episode + 1}: Reward={episode_reward:6.2f}, "
              f"Cube height={cube_height:.3f}, Contact={contact_detected}")
    
    print(f"\n🏁 Entraînement terminé")
    print(f"   Expériences collectées: {len(agent.replay_buffer)}")
    print(f"   Mises à jour effectuées: {agent.update_count}")
    
    env.close()

def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description="Démonstration du système de grasping")
    parser.add_argument("--mode", choices=["random", "trained", "training"], 
                       default="random", help="Mode de démonstration")
    parser.add_argument("--model", type=str, default=None, 
                       help="Chemin vers le modèle entraîné")
    
    args = parser.parse_args()
    
    print("🎯 SYSTÈME DE GRASPING SIMPLIFIÉ - DÉMONSTRATION")
    print("="*60)
    
    if args.mode == "random":
        demo_random_agent()
    elif args.mode == "trained":
        if args.model is None:
            print("❌ Veuillez spécifier un modèle avec --model")
            return
        demo_trained_agent(args.model)
    elif args.mode == "training":
        demo_training_progress()
    
    print("\n" + "="*60)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("="*60)

if __name__ == "__main__":
    main()