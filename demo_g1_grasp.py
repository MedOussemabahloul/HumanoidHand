#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_g1_grasp.py

Script de démonstration spécifique pour le modèle G1 avec vrais capteurs et joints.
Permet de visualiser le comportement de l'agent entraîné ou aléatoire.
"""

import os
import sys
import numpy as np
import torch
import time
import yaml
import argparse
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.g1_grasp_env import G1GraspEnv
from agents.simple_sac_agent import SimpleSACAgent

class G1GraspDemo:
    """
    Démonstration spécifique pour le modèle G1
    """
    
    def __init__(self, config_path: str = None, model_path: str = None):
        """
        Initialise la démonstration G1
        
        Args:
            config_path: Chemin vers le fichier de configuration
            model_path: Chemin vers le modèle entraîné
        """
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Configuration de l'environnement
        env_config = self.config.get("env", {})
        self.xml_path = env_config.get("xml_path", "g1_simple.xml")
        
        # Créer l'environnement
        self.env = G1GraspEnv(
            xml_path=self.xml_path,
            config=env_config
        )
        
        # Créer l'agent
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Utiliser la même configuration que l'entraînement
        agent_config = self.config.get("agent", {})
        hidden_sizes = agent_config.get("hidden_sizes", [512, 512])
        
        self.agent = SimpleSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
        
        # Charger le modèle si spécifié
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"✅ Modèle chargé: {model_path}")
            self.mode = "trained"
        else:
            print(f"⚠ Modèle non trouvé: {model_path}, utilisation de l'agent aléatoire")
            self.mode = "random"
        
        print(f"🤖 Démonstration G1 initialisée")
        print(f"   - Modèle: {self.xml_path}")
        print(f"   - Mode: {self.mode}")
        print(f"   - Observations: {obs_dim}")
        print(f"   - Actions: {act_dim}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Charge la configuration depuis un fichier YAML
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            dict: Configuration chargée
        """
        if config_path and os.path.exists(config_path):
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
    
    def run_demo(self, num_episodes: int = 5, max_steps: int = 500):
        """
        Lance la démonstration G1
        
        Args:
            num_episodes: Nombre d'épisodes de démonstration
            max_steps: Nombre maximum d'étapes par épisode
        """
        print(f"\n🎬 DÉMONSTRATION G1 - Mode: {self.mode.upper()}")
        print(f"="*60)
        
        for episode in range(num_episodes):
            print(f"\n📺 Épisode {episode + 1}/{num_episodes}")
            print("-" * 40)
            
            # Reset de l'environnement
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            has_contact = False
            has_grasp = False
            max_cube_height = 0.0
            
            for step in range(max_steps):
                # Sélectionner une action
                if self.mode == "trained":
                    action = self.agent.select_action(obs, evaluate=True)
                else:
                    action = self.env.action_space.sample()
                
                # Exécuter l'action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Mettre à jour les statistiques
                episode_reward += reward
                episode_length += 1
                
                # Suivre les événements
                if info.get('has_force_contact', False) or info.get('has_touch_contact', False):
                    has_contact = True
                
                if info.get('fingers_closed', False) and has_contact:
                    has_grasp = True
                
                max_cube_height = max(max_cube_height, info.get('cube_height', 0.0))
                
                # Afficher les informations périodiquement
                if step % 50 == 0:
                    print(f"  Step {step:3d}: Reward={reward:6.2f} | "
                          f"Height={info.get('cube_height', 0.0):.3f} | "
                          f"Contact={has_contact} | Grasp={has_grasp}")
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Résumé de l'épisode
            success = max_cube_height > 0.1
            print(f"\n📊 Résumé épisode {episode + 1}:")
            print(f"   - Reward total: {episode_reward:.2f}")
            print(f"   - Longueur: {episode_length}")
            print(f"   - Hauteur max cube: {max_cube_height:.3f}")
            print(f"   - Contact détecté: {'Oui' if has_contact else 'Non'}")
            print(f"   - Grasping réussi: {'Oui' if has_grasp else 'Non'}")
            print(f"   - Succès: {'Oui' if success else 'Non'}")
            
            # Pause entre les épisodes
            if episode < num_episodes - 1:
                print(f"\n⏳ Pause de 2 secondes...")
                time.sleep(2)
        
        # Fermer l'environnement
        self.env.close()
        print(f"\n✅ Démonstration G1 terminée")
    
    def run_interactive_demo(self):
        """
        Lance une démonstration interactive
        """
        print(f"\n🎮 DÉMONSTRATION INTERACTIVE G1")
        print(f"="*60)
        print(f"Appuyez sur 'q' pour quitter, 'r' pour reset, 's' pour step")
        
        # Reset de l'environnement
        obs, _ = self.env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            try:
                # Afficher l'état actuel
                task_info = self.env.get_task_info()
                print(f"\n📊 État actuel (Step {step_count}):")
                print(f"   - Reward: {episode_reward:.2f}")
                print(f"   - Hauteur cube: {task_info['cube_height']:.3f}")
                print(f"   - Contact: {'Oui' if task_info['has_force_contact'] or task_info['has_touch_contact'] else 'Non'}")
                print(f"   - Grasping: {'Oui' if task_info['fingers_closed'] else 'Non'}")
                
                # Attendre l'entrée utilisateur
                user_input = input("\nCommande (q/r/s): ").lower().strip()
                
                if user_input == 'q':
                    break
                elif user_input == 'r':
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    step_count = 0
                    print("🔄 Environnement reset")
                elif user_input == 's':
                    # Action de l'agent
                    if self.mode == "trained":
                        action = self.agent.select_action(obs, evaluate=True)
                    else:
                        action = self.env.action_space.sample()
                    
                    # Step
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_reward += reward
                    step_count += 1
                    obs = next_obs
                    
                    print(f"✅ Step exécuté - Reward: {reward:.2f}")
                    
                    if terminated or truncated:
                        print("🏁 Épisode terminé")
                        obs, _ = self.env.reset()
                        episode_reward = 0
                        step_count = 0
                else:
                    print("❓ Commande non reconnue")
            
            except KeyboardInterrupt:
                print("\n🛑 Interruption par l'utilisateur")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                break
        
        self.env.close()
        print(f"\n✅ Démonstration interactive terminée")

def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description="Démonstration G1 pour le grasping")
    parser.add_argument("--config", type=str, default="config/g1_grasp_config.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--model", type=str, default=None,
                       help="Chemin vers le modèle entraîné")
    parser.add_argument("--mode", type=str, choices=["random", "trained"], default="random",
                       help="Mode de démonstration")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Nombre d'épisodes de démonstration")
    parser.add_argument("--steps", type=int, default=300,
                       help="Nombre maximum d'étapes par épisode")
    parser.add_argument("--interactive", action="store_true",
                       help="Mode interactif")
    
    args = parser.parse_args()
    
    # Déterminer le modèle à charger
    model_path = args.model
    if args.mode == "trained" and not model_path:
        # Chercher un modèle dans results_g1
        results_dir = "results_g1"
        if os.path.exists(results_dir):
            model_files = [f for f in os.listdir(results_dir) if f.endswith('.pth')]
            if model_files:
                model_path = os.path.join(results_dir, model_files[-1])  # Plus récent
                print(f"🔍 Modèle trouvé automatiquement: {model_path}")
    
    # Créer la démonstration
    demo = G1GraspDemo(config_path=args.config, model_path=model_path)
    
    # Lancer la démonstration
    if args.interactive:
        demo.run_interactive_demo()
    else:
        demo.run_demo(num_episodes=args.episodes, max_steps=args.steps)

if __name__ == "__main__":
    main()