#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_g1_grasp.py

Script de test spécifique pour le modèle G1 avec vrais capteurs et joints.
Teste l'environnement G1, l'agent et la tâche individuellement.
"""

import os
import sys
import numpy as np
import torch
import time
import yaml

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.g1_grasp_env import G1GraspEnv
from agents.simple_sac_agent import SimpleSACAgent

def load_g1_config():
    """
    Charge la configuration pour le modèle G1
    """
    config_path = "config/g1_grasp_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"⚠ Configuration {config_path} non trouvée, utilisation de la configuration par défaut")
        return {
            "env": {
                "xml_path": "g1_simple.xml",
                "force_sensors": [
                    "right_thumb_force_sensor_0", "right_thumb_force_sensor_1", "right_thumb_force_sensor_2",
                    "right_index_force_sensor_0", "right_index_force_sensor_1", "right_index_force_sensor_2",
                    "right_middle_force_sensor_0", "right_middle_force_sensor_1", "right_middle_force_sensor_2",
                    "right_ring_force_sensor_0", "right_ring_force_sensor_1", "right_ring_force_sensor_2"
                ],
                "touch_sensors": [
                    "right_thumb_tip_sensor", "right_index_tip_sensor", 
                    "right_middle_tip_sensor", "right_ring_tip_sensor"
                ],
                "finger_joints": [
                    "right_thumb_joint_0", "right_thumb_joint_1",
                    "right_index_joint_0", "right_index_joint_1",
                    "right_middle_joint_0", "right_middle_joint_1",
                    "right_ring_joint_0", "right_ring_joint_1"
                ],
                "cube_body_name": "cube"
            }
        }

def test_g1_environment():
    """
    Teste l'environnement G1
    """
    print("="*60)
    print("TEST DE L'ENVIRONNEMENT G1")
    print("="*60)
    
    try:
        # Charger la configuration
        config = load_g1_config()
        env_config = config.get("env", {})
        
        # Créer l'environnement G1
        env = G1GraspEnv(
            xml_path=env_config.get("xml_path", "results/g1_combined.xml"),
            config=env_config
        )
        
        print(f"✅ Environnement G1 créé avec succès")
        print(f"   - Espace d'observation: {env.observation_space.shape}")
        print(f"   - Espace d'action: {env.action_space.shape}")
        print(f"   - Nombre de capteurs de force: {len(env.force_ids)}")
        print(f"   - Nombre de capteurs tactiles: {len(env.touch_ids)}")
        print(f"   - Nombre de joints de doigts: {len(env.finger_joint_ids)}")
        
        # Test d'un épisode simple
        obs, _ = env.reset()
        print(f"✅ Reset réussi - Observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(10):
            # Action aléatoire
            action = env.action_space.sample()
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            print(f"  Step {step}: Reward={reward:.2f}, Done={terminated or truncated}")
            print(f"    Cube height: {info['cube_height']:.3f}")
            print(f"    Force contact: {info['has_force_contact']}")
            print(f"    Touch contact: {info['has_touch_contact']}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Épisode de test terminé - Reward total: {total_reward:.2f}")
        
        # Test des informations détaillées
        task_info = env.get_task_info()
        print(f"✅ Informations de la tâche:")
        print(f"   - Force values: {task_info['force_values'][:3]}...")  # Afficher les 3 premiers
        print(f"   - Touch values: {task_info['touch_values']}")
        print(f"   - Finger positions: {task_info['finger_positions'][:3]}...")  # Afficher les 3 premiers
        
        env.close()
        print("✅ Environnement G1 fermé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'environnement G1: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_g1_agent():
    """
    Teste l'agent SAC avec l'environnement G1
    """
    print("\n" + "="*60)
    print("TEST DE L'AGENT SAC AVEC G1")
    print("="*60)
    
    try:
        # Charger la configuration
        config = load_g1_config()
        env_config = config.get("env", {})
        
        # Créer l'environnement G1
        env = G1GraspEnv(
            xml_path=env_config.get("xml_path", "g1_simple.xml"),
            config=env_config
        )
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        print(f"📊 Dimensions détectées:")
        print(f"   - Observations: {obs_dim}")
        print(f"   - Actions: {act_dim}")
        
        # Créer l'agent
        agent = SimpleSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=[256, 256],  # Réseau plus petit pour le test
            device='cpu'
        )
        
        print(f"✅ Agent SAC créé avec succès")
        
        # Test de sélection d'action
        obs = env.observation_space.sample()
        action = agent.select_action(obs)
        print(f"✅ Sélection d'action réussie - Action shape: {action.shape}")
        
        # Test d'ajout au buffer de replay
        next_obs = env.observation_space.sample()
        reward = 1.0
        done = False
        
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        print(f"✅ Expérience ajoutée au buffer - Taille: {len(agent.replay_buffer)}")
        
        # Test de mise à jour (avec peu d'expériences)
        for _ in range(10):
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
        
        agent.update(batch_size=5)
        print(f"✅ Mise à jour réussie - Nombre de mises à jour: {agent.update_count}")
        
        # Test de sauvegarde/chargement
        test_path = "test_g1_agent.pth"
        agent.save(test_path)
        print(f"✅ Sauvegarde réussie: {test_path}")
        
        # Créer un nouvel agent et charger
        new_agent = SimpleSACAgent(obs_dim, act_dim, device='cpu')
        new_agent.load(test_path)
        print(f"✅ Chargement réussi")
        
        # Nettoyer
        os.remove(test_path)
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'agent G1: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_g1_integration():
    """
    Teste l'intégration complète G1
    """
    print("\n" + "="*60)
    print("TEST D'INTÉGRATION G1")
    print("="*60)
    
    try:
        # Charger la configuration
        config = load_g1_config()
        env_config = config.get("env", {})
        
        # Créer l'environnement G1
        env = G1GraspEnv(
            xml_path=env_config.get("xml_path", "g1_simple.xml"),
            config=env_config
        )
        
        # Créer l'agent
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        agent = SimpleSACAgent(obs_dim, act_dim, hidden_sizes=[128, 128], device='cpu')
        
        print(f"✅ Intégration G1 réussie - Tous les composants créés")
        
        # Test d'un épisode complet
        obs, _ = env.reset()
        
        episode_reward = 0
        for step in range(20):
            # Action de l'agent
            action = agent.select_action(obs)
            
            # Step dans l'environnement
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Stocker l'expérience
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            # Afficher les informations périodiquement
            if step % 5 == 0:
                print(f"  Step {step}: Reward={reward:.2f}, Cube height={info['cube_height']:.3f}")
                print(f"    Force contact: {info['has_force_contact']}, Touch contact: {info['has_touch_contact']}")
            
            if done:
                break
        
        print(f"✅ Épisode d'intégration G1 terminé - Reward: {episode_reward:.2f}")
        
        # Test de mise à jour
        if len(agent.replay_buffer) >= 10:
            agent.update(batch_size=5)
            print(f"✅ Mise à jour d'intégration G1 réussie")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test d'intégration G1: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Fonction principale de test G1
    """
    print("🤖 DÉBUT DES TESTS DU SYSTÈME G1")
    print("="*80)
    
    # Tests individuels
    tests = [
        ("Environnement G1", test_g1_environment),
        ("Agent SAC avec G1", test_g1_agent),
        ("Intégration G1", test_g1_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erreur critique lors du test {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS G1")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name:25} : {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS G1 SONT PASSÉS !")
        print("Le système G1 est prêt à être utilisé.")
    else:
        print("⚠ CERTAINS TESTS G1 ONT ÉCHOUÉ")
        print("Vérifiez les erreurs ci-dessus avant de continuer.")
    
    print("="*80)

if __name__ == "__main__":
    main()