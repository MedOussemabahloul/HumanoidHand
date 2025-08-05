#!/usr/bin/env python3
"""
Script d'entraînement simplifié pour la saisie G1
Utilise SAC avec curriculum learning et enregistrement vidéo
"""

import os
import sys
import argparse
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch non disponible - mode simulation")

import time
from pathlib import Path
from datetime import datetime
import json

# Ajouter les modules locaux au path
sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')
sys.path.append('./utils')

if HAS_TORCH:
    from envs.simple_grasp_env import SimpleGraspEnv
    from agents.improved_sac_agent import ImprovedSACAgent
from utils.video_recorder import VideoRecorder

class GraspTrainer:
    """Entraîneur pour la tâche de saisie G1"""
    
    def __init__(self, config):
        self.config = config
        
        # Créer les dossiers de sortie
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        if not HAS_TORCH:
            print("❌ PyTorch requis pour l'entraînement")
            print("💡 Installation: pip install torch")
            return
        
        # Initialiser l'environnement
        print("🤖 Initialisation de l'environnement...")
        self.env = SimpleGraspEnv(
            xml_path=config['model_path'],
            max_episode_steps=config['max_episode_steps'],
            curriculum_level=config['curriculum_level']
        )
        
        # Initialiser l'agent SAC
        print("🧠 Initialisation de l'agent SAC...")
        self.agent = ImprovedSACAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            lr=config['learning_rate'],
            hidden_sizes=config['hidden_sizes'],
            buffer_size=config['buffer_size']
        )
        
        # Enregistreur vidéo
        self.video_recorder = VideoRecorder(
            output_dir=self.output_dir / "videos",
            fps=config['video_fps']
        )
        
        # Métriques d'entraînement
        self.episode_rewards = []
        self.episode_lengths = []
        
        print("✅ Entraîneur initialisé")
    
    def train(self):
        """Lance l'entraînement"""
        if not HAS_TORCH:
            print("❌ Impossible de lancer l'entraînement sans PyTorch")
            return
            
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        for episode in range(total_episodes):
            # Reset de l'environnement
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Épisode
            done = False
            while not done:
                # Sélection d'action
                action = self.agent.select_action(obs)
                
                # Étape d'environnement
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Stocker la transition
                self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Mise à jour des métriques
                episode_reward += reward
                episode_length += 1
                
                obs = next_obs
            
            # Entraînement de l'agent
            if len(self.agent.replay_buffer) > self.config['batch_size']:
                for _ in range(self.config['updates_per_episode']):
                    self.agent.update(self.config['batch_size'])
            
            # Enregistrer les métriques
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Affichage des progrès
            if (episode + 1) % self.config['log_interval'] == 0:
                self._log_progress(episode + 1, total_episodes, start_time)
            
            # Enregistrement vidéo périodique
            if (episode + 1) % self.config['video_interval'] == 0:
                try:
                    self.video_recorder.record_episode(self.env, self.agent)
                except Exception as e:
                    print(f"⚠️  Erreur vidéo: {e}")
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ")
        print(f"   Durée totale: {total_time/3600:.1f}h")
        print(f"   Récompense moyenne finale: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # Sauvegarde finale
        self._save_final_results()
    
    def _log_progress(self, episode, total_episodes, start_time):
        """Affiche les progrès d'entraînement"""
        recent_rewards = self.episode_rewards[-self.config['log_interval']:]
        avg_reward = np.mean(recent_rewards)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 Épisode {episode}/{total_episodes}")
        print(f"   Récompense: {avg_reward:.2f}")
        print(f"   Buffer: {len(self.agent.replay_buffer)}")
        print(f"   Temps: {elapsed_time/60:.1f}min")
    
    def _save_final_results(self):
        """Sauvegarde les résultats finaux"""
        # Sauvegarder le modèle final
        final_model_path = self.output_dir / "models" / "final_model.pth"
        self.agent.save(final_model_path)
        
        # Sauvegarder les métriques
        final_metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "avg_reward_last_100": float(np.mean(self.episode_rewards[-100:])),
            "total_episodes": len(self.episode_rewards)
        }
        
        final_metrics_path = self.output_dir / "logs" / "final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✅ Résultats sauvegardés dans {self.output_dir}")

def load_config():
    """Charge la configuration d'entraînement"""
    return {
        # Environnement
        'model_path': 'results/g1_combined.xml',
        'max_episode_steps': 500,
        'curriculum_level': 1,
        
        # Entraînement
        'total_episodes': 1000,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 100000,
        'updates_per_episode': 1,
        'hidden_sizes': [256, 256],
        
        # Logging
        'log_interval': 50,
        'video_interval': 100,
        'video_fps': 30,
        
        # Sortie
        'output_dir': 'training_results'
    }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement de saisie G1')
    parser.add_argument('--episodes', type=int, default=1000, help='Nombre d\'épisodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Taux d\'apprentissage')
    parser.add_argument('--output', type=str, default='training_results', help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Charger et modifier la configuration
    config = load_config()
    config['total_episodes'] = args.episodes
    config['learning_rate'] = args.lr
    config['output_dir'] = args.output
    
    print("🤖 ENTRAÎNEMENT DE SAISIE G1")
    print("=" * 50)
    print(f"Épisodes: {config['total_episodes']}")
    print(f"Taux d'apprentissage: {config['learning_rate']}")
    print(f"Dossier de sortie: {config['output_dir']}")
    
    # Vérifier que le modèle existe
    if not Path(config['model_path']).exists():
        print(f"❌ Modèle non trouvé: {config['model_path']}")
        print("💡 Placez le modèle g1_combined.xml dans le dossier results/")
        return
    
    # Créer et lancer l'entraîneur
    try:
        trainer = GraspTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
