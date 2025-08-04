#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_simple_grasp.py

Script de test simple pour vérifier le fonctionnement du système de grasping.
Teste l'environnement, l'agent et la tâche individuellement.
"""

import os
import sys
import numpy as np
import torch
import time

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.simple_grasp_env import SimpleGraspEnv
from agents.simple_sac_agent import SimpleSACAgent
from tasks.grasp.simple_grasp_task import SimpleGraspTask

def test_environment():
    """
    Teste l'environnement de simulation
    """
    print("="*50)
    print("TEST DE L'ENVIRONNEMENT")
    print("="*50)
    
    try:
        # Créer l'environnement
        env = SimpleGraspEnv(
            xml_path="assets/scenes/complete_scene.xml",
            render_mode=None,
            config={
                "touch_sensors": ["touch1_sensor", "touch2_sensor"],
                "cube_body_name": "cube",
                "max_steps_per_episode": 100
            }
        )
        
        print(f"✓ Environnement créé avec succès")
        print(f"  - Espace d'observation: {env.observation_space.shape}")
        print(f"  - Espace d'action: {env.action_space.shape}")
        print(f"  - Nombre de capteurs tactiles: {len(env.touch_ids)}")
        
        # Test d'un épisode simple
        obs, _ = env.reset()
        print(f"✓ Reset réussi - Observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(10):
            # Action aléatoire
            action = env.action_space.sample()
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            print(f"  Step {step}: Reward={reward:.2f}, Done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        print(f"✓ Épisode de test terminé - Reward total: {total_reward:.2f}")
        
        # Test du rendu
        frame = env.render(mode="rgb_array")
        if frame is not None:
            print(f"✓ Rendu réussi - Frame shape: {frame.shape}")
        else:
            print("⚠ Rendu non disponible")
        
        env.close()
        print("✓ Environnement fermé")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test de l'environnement: {e}")
        return False

def test_agent():
    """
    Teste l'agent SAC
    """
    print("\n" + "="*50)
    print("TEST DE L'AGENT SAC")
    print("="*50)
    
    try:
        # Créer un environnement pour obtenir les dimensions
        env = SimpleGraspEnv(config={"touch_sensors": ["touch1_sensor", "touch2_sensor"]})
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # Créer l'agent
        agent = SimpleSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=[64, 64],  # Réseau plus petit pour le test
            device='cpu'
        )
        
        print(f"✓ Agent créé avec succès")
        print(f"  - Dimension d'observation: {obs_dim}")
        print(f"  - Dimension d'action: {act_dim}")
        
        # Test de sélection d'action
        obs = env.observation_space.sample()
        action = agent.select_action(obs)
        print(f"✓ Sélection d'action réussie - Action shape: {action.shape}")
        
        # Test d'ajout au buffer de replay
        next_obs = env.observation_space.sample()
        reward = 1.0
        done = False
        
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        print(f"✓ Expérience ajoutée au buffer - Taille: {len(agent.replay_buffer)}")
        
        # Test de mise à jour (avec peu d'expériences)
        for _ in range(10):
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
        
        agent.update(batch_size=5)
        print(f"✓ Mise à jour réussie - Nombre de mises à jour: {agent.update_count}")
        
        # Test de sauvegarde/chargement
        test_path = "test_agent.pth"
        agent.save(test_path)
        print(f"✓ Sauvegarde réussie: {test_path}")
        
        # Créer un nouvel agent et charger
        new_agent = SimpleSACAgent(obs_dim, act_dim, device='cpu')
        new_agent.load(test_path)
        print(f"✓ Chargement réussi")
        
        # Nettoyer
        os.remove(test_path)
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test de l'agent: {e}")
        return False

def test_task():
    """
    Teste la tâche de grasping
    """
    print("\n" + "="*50)
    print("TEST DE LA TÂCHE DE GRASPING")
    print("="*50)
    
    try:
        # Créer l'environnement
        env = SimpleGraspEnv(config={"touch_sensors": ["touch1_sensor", "touch2_sensor"]})
        
        # Créer la tâche
        task_config = {
            "cube_body_name": "cube",
            "max_steps_per_episode": 100,
            "touch_sensors": ["touch1_sensor", "touch2_sensor"],
            "record_video": False
        }
        
        task = SimpleGraspTask(env.model, env.data, task_config)
        
        print(f"✓ Tâche créée avec succès")
        print(f"  - Dimension d'observation: {task.obs_dim}")
        print(f"  - Dimension d'action: {task.act_dim}")
        
        # Test d'un épisode
        obs = task.reset()
        print(f"✓ Reset de la tâche réussi - Observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(10):
            # Action aléatoire
            action = np.random.uniform(-1, 1, task.act_dim)
            
            # Step
            obs, reward, done, info = task.step(action)
            
            total_reward += reward
            
            print(f"  Step {step}: Reward={reward:.2f}, Done={done}")
            
            if done:
                break
        
        print(f"✓ Épisode de test terminé - Reward total: {total_reward:.2f}")
        
        # Test des informations de la tâche
        task_info = task.get_task_info()
        print(f"✓ Informations de la tâche: {task_info}")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test de la tâche: {e}")
        return False

def test_integration():
    """
    Teste l'intégration complète
    """
    print("\n" + "="*50)
    print("TEST D'INTÉGRATION")
    print("="*50)
    
    try:
        # Créer l'environnement
        env = SimpleGraspEnv(config={"touch_sensors": ["touch1_sensor", "touch2_sensor"]})
        
        # Créer l'agent
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        agent = SimpleSACAgent(obs_dim, act_dim, hidden_sizes=[64, 64], device='cpu')
        
        # Créer la tâche
        task_config = {
            "cube_body_name": "cube",
            "max_steps_per_episode": 50,
            "touch_sensors": ["touch1_sensor", "touch2_sensor"],
            "record_video": False
        }
        task = SimpleGraspTask(env.model, env.data, task_config)
        
        print(f"✓ Intégration réussie - Tous les composants créés")
        
        # Test d'un épisode complet
        obs, _ = env.reset()
        obs = task.reset()
        
        episode_reward = 0
        for step in range(20):
            # Action de l'agent
            action = agent.select_action(obs)
            
            # Step dans l'environnement
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Step dans la tâche
            task_obs, task_reward, task_done, task_info = task.step(action)
            
            # Stocker l'expérience
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        print(f"✓ Épisode d'intégration terminé - Reward: {episode_reward:.2f}")
        
        # Test de mise à jour
        if len(agent.replay_buffer) >= 10:
            agent.update(batch_size=5)
            print(f"✓ Mise à jour d'intégration réussie")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test d'intégration: {e}")
        return False

def main():
    """
    Fonction principale de test
    """
    print("DÉBUT DES TESTS DU SYSTÈME DE GRASPING")
    print("="*60)
    
    # Tests individuels
    tests = [
        ("Environnement", test_environment),
        ("Agent SAC", test_agent),
        ("Tâche de grasping", test_task),
        ("Intégration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Erreur critique lors du test {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSÉ" if passed else "✗ ÉCHOUÉ"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        print("Le système de grasping est prêt à être utilisé.")
    else:
        print("⚠ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez les erreurs ci-dessus avant de continuer.")
    
    print("="*60)

if __name__ == "__main__":
    main()