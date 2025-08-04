#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_simple_grasp.py

Script de training simplifié pour l'apprentissage du grasping avec SAC.
Génère des vidéos de simulation et affiche les récompenses en temps réel.
"""

import os
import sys
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import yaml
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.simple_grasp_env import SimpleGraspEnv
from agents.simple_sac_agent import SimpleSACAgent
from tasks.grasp.simple_grasp_task import SimpleGraspTask

class SimpleGraspTrainer:
    """
    Entraîneur simplifié pour l'apprentissage du grasping
    """
    def __init__(self, config_path: str = None):
        """
        Initialise l'entraîneur
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Créer les répertoires de sortie
        self.output_dir = Path(self.config.get("output_dir", "results"))
        self.video_dir = self.output_dir / "videos"
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.video_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")
        
        # Initialiser l'environnement
        self.env = self._create_environment()
        
        # Initialiser l'agent
        self.agent = self._create_agent()
        
        # Statistiques d'entraînement
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Paramètres d'entraînement
        self.max_episodes = self.config.get("max_episodes", 1000)
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 1000)
        self.update_frequency = self.config.get("update_frequency", 1)
        self.batch_size = self.config.get("batch_size", 256)
        self.save_frequency = self.config.get("save_frequency", 100)
        self.eval_frequency = self.config.get("eval_frequency", 50)
        
        print("Entraîneur initialisé avec succès!")
    
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
                config = yaml.safe_load(f)
        else:
            # Configuration par défaut
            config = {
                "env": {
                    "xml_path": "assets/scenes/complete_scene.xml",
                    "render_mode": None,
                    "width": 640,
                    "height": 480,
                    "max_steps_per_episode": 1000,
                    "touch_sensors": ["touch1_sensor", "touch2_sensor"],
                    "cube_body_name": "cube"
                },
                "agent": {
                    "hidden_sizes": [256, 256],
                    "lr": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "alpha": 0.2
                },
                "training": {
                    "max_episodes": 1000,
                    "max_steps_per_episode": 1000,
                    "update_frequency": 1,
                    "batch_size": 256,
                    "save_frequency": 100,
                    "eval_frequency": 50
                },
                "output_dir": "results"
            }
        
        return config
    
    def _create_environment(self) -> SimpleGraspEnv:
        """
        Crée l'environnement de simulation
        
        Returns:
            SimpleGraspEnv: Environnement créé
        """
        env_config = self.config.get("env", {})
        
        env = SimpleGraspEnv(
            xml_path=env_config.get("xml_path", "assets/scenes/complete_scene.xml"),
            render_mode=env_config.get("render_mode"),
            width=env_config.get("width", 640),
            height=env_config.get("height", 480),
            config=env_config
        )
        
        print(f"Environnement créé - Obs dim: {env.observation_space.shape}, Act dim: {env.action_space.shape}")
        return env
    
    def _create_agent(self) -> SimpleSACAgent:
        """
        Crée l'agent SAC
        
        Returns:
            SimpleSACAgent: Agent créé
        """
        agent_config = self.config.get("agent", {})
        
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        
        agent = SimpleSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=agent_config.get("hidden_sizes", [256, 256]),
            lr=agent_config.get("lr", 3e-4),
            gamma=agent_config.get("gamma", 0.99),
            tau=agent_config.get("tau", 0.005),
            alpha=agent_config.get("alpha", 0.2),
            device=self.device
        )
        
        print(f"Agent SAC créé - Obs dim: {obs_dim}, Act dim: {act_dim}")
        return agent
    
    def train(self):
        """
        Lance l'entraînement
        """
        print("Début de l'entraînement...")
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            episode_reward = 0
            episode_length = 0
            obs, _ = self.env.reset()
            
            # Variables pour l'enregistrement vidéo
            frames = []
            record_video = False  # Désactivé pour éviter les erreurs de rendu
            
            for step in range(self.max_steps_per_episode):
                # Sélectionner une action
                action = self.agent.select_action(obs)
                
                # Exécuter l'action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Stocker l'expérience
                done = terminated or truncated
                self.agent.replay_buffer.push(obs, action, reward, next_obs, done)
                
                # Mettre à jour l'agent
                if step % self.update_frequency == 0:
                    self.agent.update(batch_size=self.batch_size)
                
                # Enregistrer la frame si nécessaire (désactivé)
                # if record_video:
                #     frame = self.env.render(mode="rgb_array")
                #     if frame is not None:
                #         frames.append(frame)
                
                # Mettre à jour les statistiques
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                # Vérifier si l'épisode est terminé
                if done:
                    break
            
            # Mettre à jour les statistiques globales
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_episodes += 1
            
            # Vérifier le succès (cube soulevé)
            if info.get('cube_height', 0) > 0.1:
                self.success_count += 1
            
            # Afficher les statistiques
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / max(1, self.total_episodes)
                print(f"Episode {episode}/{self.max_episodes} - "
                      f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) - "
                      f"Length: {episode_length} - "
                      f"Success Rate: {success_rate:.2%}")
            
            # Sauvegarder la vidéo si nécessaire
            if record_video and frames:
                self._save_video(frames, episode)
            
            # Sauvegarder le modèle périodiquement
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_model(episode)
        
        # Sauvegarder le modèle final
        self._save_model("final")
        
        # Afficher les statistiques finales
        self._print_final_stats()
        
        # Sauvegarder les graphiques
        self._save_plots()
        
        print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes")
    
    def _save_video(self, frames: list, episode: int):
        """
        Sauvegarde une vidéo de l'épisode
        
        Args:
            frames: Liste des frames
            episode: Numéro de l'épisode
        """
        if not frames:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grasp_episode_{episode}_{timestamp}.mp4"
        filepath = self.video_dir / filename
        
        # Paramètres de la vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        height, width = frames[0].shape[:2]
        
        # Créer le writer vidéo
        out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        
        # Écrire toutes les frames
        for frame in frames:
            # Convertir RGB vers BGR pour OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Vidéo sauvegardée: {filepath}")
    
    def _save_model(self, episode):
        """
        Sauvegarde le modèle de l'agent
        
        Args:
            episode: Numéro de l'épisode ou "final"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sac_grasp_episode_{episode}_{timestamp}.pth"
        filepath = self.model_dir / filename
        
        self.agent.save(str(filepath))
        print(f"Modèle sauvegardé: {filepath}")
    
    def _print_final_stats(self):
        """
        Affiche les statistiques finales de l'entraînement
        """
        print("\n" + "="*50)
        print("STATISTIQUES FINALES")
        print("="*50)
        print(f"Nombre total d'épisodes: {self.total_episodes}")
        print(f"Taux de succès: {self.success_count / max(1, self.total_episodes):.2%}")
        print(f"Récompense moyenne: {np.mean(self.episode_rewards):.2f}")
        print(f"Récompense maximale: {np.max(self.episode_rewards):.2f}")
        print(f"Longueur moyenne des épisodes: {np.mean(self.episode_lengths):.2f}")
        print(f"Nombre de mises à jour: {self.agent.update_count}")
        print("="*50)
    
    def _save_plots(self):
        """
        Sauvegarde les graphiques de l'entraînement
        """
        # Graphique des récompenses
        plt.figure(figsize=(12, 8))
        
        # Récompenses par épisode
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Récompenses par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense")
        plt.grid(True)
        
        # Récompense moyenne mobile
        plt.subplot(2, 2, 2)
        window_size = min(100, len(self.episode_rewards) // 10)
        if window_size > 0:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.episode_rewards)), moving_avg)
        plt.title(f"Récompense moyenne mobile (fenêtre {window_size})")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense moyenne")
        plt.grid(True)
        
        # Longueur des épisodes
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_lengths)
        plt.title("Longueur des épisodes")
        plt.xlabel("Épisode")
        plt.ylabel("Nombre d'étapes")
        plt.grid(True)
        
        # Taux de succès
        plt.subplot(2, 2, 4)
        success_rates = []
        for i in range(1, len(self.episode_rewards) + 1):
            success_count = sum(1 for j in range(i) if self.episode_rewards[j] > 50)  # Seuil de succès
            success_rates.append(success_count / i)
        plt.plot(success_rates)
        plt.title("Taux de succès cumulatif")
        plt.xlabel("Épisode")
        plt.ylabel("Taux de succès")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plot_path = self.log_dir / f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphiques sauvegardés: {plot_path}")

def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description="Entraînement simplifié pour le grasping")
    parser.add_argument("--config", type=str, default=None, help="Chemin vers le fichier de configuration")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--output_dir", type=str, default="results", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    # Créer l'entraîneur
    trainer = SimpleGraspTrainer(config_path=args.config)
    
    # Modifier le nombre d'épisodes si spécifié
    if args.episodes != 1000:
        trainer.max_episodes = args.episodes
    
    # Modifier le répertoire de sortie si spécifié
    if args.output_dir != "results":
        trainer.output_dir = Path(args.output_dir)
        trainer.video_dir = trainer.output_dir / "videos"
        trainer.model_dir = trainer.output_dir / "models"
        trainer.log_dir = trainer.output_dir / "logs"
        
        for dir_path in [trainer.output_dir, trainer.video_dir, trainer.model_dir, trainer.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Lancer l'entraînement
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        trainer._save_model("interrupted")
        trainer._print_final_stats()
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        trainer._save_model("error")
        raise

if __name__ == "__main__":
    main()