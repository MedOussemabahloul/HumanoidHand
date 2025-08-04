#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_g1_grasp.py

Script d'entraînement spécifique pour le modèle G1 avec vrais capteurs et joints.
Utilise l'environnement G1GraspEnv et l'agent SAC pour apprendre le grasping.
"""

import os
import sys
import numpy as np
import torch
import time
import yaml
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.g1_grasp_env import G1GraspEnv
from agents.simple_sac_agent import SimpleSACAgent

class G1GraspTrainer:
    """
    Entraîneur spécifique pour le modèle G1
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialise l'entraîneur G1
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Créer les répertoires de sortie
        self.output_dir = self.config.get("output_dir", "results_g1")
        self.video_path = self.config.get("task", {}).get("video_path", "videos_g1")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)
        
        # Configuration de l'environnement
        env_config = self.config.get("env", {})
        self.xml_path = env_config.get("xml_path", "g1_simple.xml")
        self.max_steps_per_episode = env_config.get("max_steps_per_episode", 1000)
        
        # Configuration de l'agent
        agent_config = self.config.get("agent", {})
        self.hidden_sizes = agent_config.get("hidden_sizes", [512, 512])
        self.lr = float(agent_config.get("lr", 3e-4))
        self.gamma = float(agent_config.get("gamma", 0.99))
        self.tau = float(agent_config.get("tau", 0.005))
        self.alpha = float(agent_config.get("alpha", 0.2))
        
        # Configuration de l'entraînement
        training_config = self.config.get("training", {})
        self.max_episodes = training_config.get("max_episodes", 1000)
        self.update_frequency = training_config.get("update_frequency", 1)
        self.batch_size = training_config.get("batch_size", 256)
        self.save_frequency = training_config.get("save_frequency", 100)
        self.eval_frequency = training_config.get("eval_frequency", 50)
        
        # Configuration de la tâche
        task_config = self.config.get("task", {})
        self.contact_reward = task_config.get("contact_reward", 10.0)
        self.grasp_reward = task_config.get("grasp_reward", 50.0)
        self.lift_reward_weight = task_config.get("lift_reward_weight", 1.0)
        self.record_video = False  # Désactivé pour éviter les erreurs de rendu
        
        # Configuration du système
        system_config = self.config.get("system", {})
        self.use_gpu = system_config.get("use_gpu", True) and torch.cuda.is_available()
        self.verbose = system_config.get("verbose", 1)
        
        # Créer l'environnement et l'agent
        self.env = self._create_environment()
        self.agent = self._create_agent()
        
        # Statistiques d'entraînement
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.contact_rates = []
        self.grasp_rates = []
        
        print(f"🤖 Entraîneur G1 initialisé")
        print(f"   - Modèle: {self.xml_path}")
        print(f"   - Épisodes: {self.max_episodes}")
        print(f"   - GPU: {'Oui' if self.use_gpu else 'Non'}")
        print(f"   - Sortie: {self.output_dir}")
    
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
                    "max_steps_per_episode": 1000,
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
                },
                "agent": {
                    "hidden_sizes": [512, 512],
                    "lr": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "alpha": 0.2
                },
                "training": {
                    "max_episodes": 1000,
                    "update_frequency": 1,
                    "batch_size": 256,
                    "save_frequency": 100,
                    "eval_frequency": 50
                },
                "task": {
                    "contact_reward": 10.0,
                    "grasp_reward": 50.0,
                    "lift_reward_weight": 1.0,
                    "record_video": True,
                    "video_path": "videos_g1"
                },
                "output_dir": "results_g1",
                "system": {
                    "use_gpu": True,
                    "verbose": 1
                }
            }
    
    def _create_environment(self) -> G1GraspEnv:
        """
        Crée l'environnement G1
        
        Returns:
            G1GraspEnv: Environnement créé
        """
        env_config = self.config.get("env", {})
        return G1GraspEnv(
            xml_path=self.xml_path,
            config=env_config
        )
    
    def _create_agent(self) -> SimpleSACAgent:
        """
        Crée l'agent SAC
        
        Returns:
            SimpleSACAgent: Agent créé
        """
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        device = 'cuda' if self.use_gpu else 'cpu'
        
        return SimpleSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=self.hidden_sizes,
            lr=self.lr,
            gamma=self.gamma,
            tau=self.tau,
            alpha=self.alpha,
            device=device
        )
    
    def train(self):
        """
        Lance l'entraînement G1
        """
        print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT G1")
        print(f"="*60)
        
        start_time = time.time()
        
        for episode in tqdm(range(self.max_episodes), desc="Épisodes G1"):
            # Reset de l'environnement
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            has_contact = False
            has_grasp = False
            max_cube_height = 0.0
            
            # Variables pour l'enregistrement vidéo
            frames = []
            record_video = self.record_video and (episode % self.eval_frequency == 0)
            
            for step in range(self.max_steps_per_episode):
                # Sélectionner une action
                action = self.agent.select_action(obs, evaluate=False)
                
                # Exécuter l'action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Stocker l'expérience
                done = terminated or truncated
                self.agent.replay_buffer.push(obs, action, reward, next_obs, done)
                
                # Mettre à jour l'agent
                if len(self.agent.replay_buffer) > self.batch_size and step % self.update_frequency == 0:
                    self.agent.update(batch_size=self.batch_size)
                
                # Enregistrer la frame si nécessaire
                if record_video:
                    frame = self.env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)
                
                # Mettre à jour les statistiques
                episode_reward += reward
                episode_length += 1
                
                # Suivre les événements
                if info.get('has_force_contact', False) or info.get('has_touch_contact', False):
                    has_contact = True
                
                if info.get('fingers_closed', False) and has_contact:
                    has_grasp = True
                
                max_cube_height = max(max_cube_height, info.get('cube_height', 0.0))
                
                obs = next_obs
                
                if done:
                    break
            
            # Sauvegarder la vidéo si nécessaire
            if record_video and frames:
                self._save_video(frames, episode)
            
            # Sauvegarder le modèle périodiquement
            if episode % self.save_frequency == 0:
                self._save_model(episode)
            
            # Enregistrer les statistiques
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rates.append(1.0 if max_cube_height > 0.1 else 0.0)
            self.contact_rates.append(1.0 if has_contact else 0.0)
            self.grasp_rates.append(1.0 if has_grasp else 0.0)
            
            # Afficher les progrès
            if self.verbose >= 1 and episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_success = np.mean(self.success_rates[-10:])
                avg_contact = np.mean(self.contact_rates[-10:])
                avg_grasp = np.mean(self.grasp_rates[-10:])
                
                print(f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | "
                      f"Success: {avg_success:.2f} | Contact: {avg_contact:.2f} | "
                      f"Grasp: {avg_grasp:.2f} | Height: {max_cube_height:.3f}")
        
        # Sauvegarder le modèle final
        self._save_model("final")
        
        # Afficher les statistiques finales
        self._print_final_stats()
        
        # Sauvegarder les graphiques
        self._save_plots()
        
        # Fermer l'environnement
        self.env.close()
        
        training_time = time.time() - start_time
        print(f"\n✅ Entraînement G1 terminé en {training_time:.1f} secondes")
    
    def _save_video(self, frames: list, episode: int):
        """
        Sauvegarde une vidéo de l'épisode
        
        Args:
            frames: Liste des frames
            episode: Numéro de l'épisode
        """
        try:
            import cv2
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_path, f"g1_episode_{episode}_{timestamp}.mp4")
            
            if frames:
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                for frame in frames:
                    # Convertir RGB vers BGR pour OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"📹 Vidéo sauvegardée: {video_path}")
        except ImportError:
            print("⚠ OpenCV non disponible, vidéo non sauvegardée")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde de la vidéo: {e}")
    
    def _save_model(self, episode):
        """
        Sauvegarde le modèle de l'agent
        
        Args:
            episode: Numéro de l'épisode ou "final"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f"g1_sac_episode_{episode}_{timestamp}.pth")
        
        try:
            self.agent.save(model_path)
            if self.verbose >= 1:
                print(f"💾 Modèle sauvegardé: {model_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du modèle: {e}")
    
    def _print_final_stats(self):
        """
        Affiche les statistiques finales de l'entraînement
        """
        print(f"\n📊 STATISTIQUES FINALES G1")
        print(f"="*60)
        print(f"Épisodes totaux: {len(self.episode_rewards)}")
        print(f"Reward moyen: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        print(f"Longueur moyenne: {np.mean(self.episode_lengths):.1f} ± {np.std(self.episode_lengths):.1f}")
        print(f"Taux de succès: {np.mean(self.success_rates):.2f}")
        print(f"Taux de contact: {np.mean(self.contact_rates):.2f}")
        print(f"Taux de grasping: {np.mean(self.grasp_rates):.2f}")
        
        # Statistiques des 100 derniers épisodes
        if len(self.episode_rewards) >= 100:
            recent_rewards = self.episode_rewards[-100:]
            recent_success = self.success_rates[-100:]
            print(f"\n📈 100 derniers épisodes:")
            print(f"Reward moyen: {np.mean(recent_rewards):.2f}")
            print(f"Taux de succès: {np.mean(recent_success):.2f}")
    
    def _save_plots(self):
        """
        Sauvegarde les graphiques de l'entraînement
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Rewards
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Rewards par épisode')
            axes[0, 0].set_xlabel('Épisode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # Longueurs d'épisode
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Longueur par épisode')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Longueur')
            axes[0, 1].grid(True)
            
            # Taux de succès
            window = 50
            if len(self.success_rates) >= window:
                success_smooth = np.convolve(self.success_rates, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(success_smooth)
                axes[1, 0].set_title(f'Taux de succès (moyenne sur {window} épisodes)')
                axes[1, 0].set_xlabel('Épisode')
                axes[1, 0].set_ylabel('Taux de succès')
                axes[1, 0].grid(True)
            
            # Taux de contact et grasping
            if len(self.contact_rates) >= window:
                contact_smooth = np.convolve(self.contact_rates, np.ones(window)/window, mode='valid')
                grasp_smooth = np.convolve(self.grasp_rates, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(contact_smooth, label='Contact')
                axes[1, 1].plot(grasp_smooth, label='Grasping')
                axes[1, 1].set_title(f'Taux de contact et grasping (moyenne sur {window} épisodes)')
                axes[1, 1].set_xlabel('Épisode')
                axes[1, 1].set_ylabel('Taux')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.output_dir, f"g1_training_plots_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graphiques sauvegardés: {plot_path}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des graphiques: {e}")

def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description="Entraînement G1 pour le grasping")
    parser.add_argument("--config", type=str, default="config/g1_grasp_config.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Nombre d'épisodes d'entraînement")
    
    args = parser.parse_args()
    
    # Créer l'entraîneur
    trainer = G1GraspTrainer(config_path=args.config)
    
    # Modifier le nombre d'épisodes si spécifié
    if args.episodes is not None:
        trainer.max_episodes = args.episodes
    
    # Lancer l'entraînement
    trainer.train()

if __name__ == "__main__":
    main()