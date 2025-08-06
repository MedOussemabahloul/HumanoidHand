
#!/usr/bin/env python3
"""
Script d'entraînement simplifié pour la saisie G1
Utilise SAC avec curriculum learning et enregistrement vidéo
Auteur: Assistant IA
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Ajouter le projet au path
sys.path.append('/home/oussema/Documents/project/')
sys.path.append('/home/oussema/Documents/project/envs')
sys.path.append('/home/oussema/Documents/project/agents')
sys.path.append('/home/oussema/Documents/project/utils')

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
        self.success_rate = []
        self.contact_rate = []
        self.training_metrics = []
        
        # Curriculum learning
        self.curriculum_threshold = config['curriculum_threshold']
        self.episodes_per_level = config['episodes_per_level']
        
        print("✅ Entraîneur initialisé")
        print(f"   Espace d'observation: {self.env.observation_space.shape}")
        print(f"   Espace d'action: {self.env.action_space.shape}")
        print(f"   Niveau curriculum: {config['curriculum_level']}")
    
    def train(self):
        """Lance l'entraînement"""
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        # Variables pour le curriculum learning
        current_level = self.config['curriculum_level']
        episodes_at_level = 0
        level_successes = []
        
        for episode in range(total_episodes):
            episode_start_time = time.time()
            
            # Reset de l'environnement
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            contact_made = False
            success = False
            
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
                episode_reward = reward
                episode_length = 1
                contact_made = contact_made or info.get('contact', False)
                success = terminated and info.get('cube_height', 0) > 0.1
                
                obs = next_obs
            
            # Entraînement de l'agent
            if len(self.agent.replay_buffer) > self.config['batch_size']:
                for _ in range(self.config['updates_per_episode']):
                    training_info = self.agent.update(self.config['batch_size'])
                    if training_info:
                        self.training_metrics.append(training_info)
            
            # Enregistrer les métriques
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            level_successes.append(success)
            
            # Curriculum learning
            episodes_at_level = 1
            if episodes_at_level >= self.episodes_per_level:
                success_rate = np.mean(level_successes[-self.episodes_per_level:])
                if success_rate >= self.curriculum_threshold and current_level < 3:
                    current_level = 1
                    self.env.curriculum_level = current_level
                    episodes_at_level = 0
                    print(f"🎓 Passage au niveau curriculum {current_level}")
                elif episodes_at_level >= self.episodes_per_level * 2:
                    episodes_at_level = 0
            
            # Affichage des progrès
            if (episode + 1) % self.config['log_interval'] == 0:
                self._log_progress(episode + 1, total_episodes, start_time)
            
            # Sauvegarde périodique
            if (episode + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(episode + 1)
            
            # Enregistrement vidéo périodique
            if (episode + 1) % self.config['video_interval'] == 0:
                self._record_evaluation_video(episode + 1)
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ")
        print(f"   Durée totale: {total_time/3600:.1f}h")
        print(f"   Épisodes: {total_episodes}")
        print(f"   Récompense moyenne finale: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # Sauvegarde finale
        self._save_final_results()
        
        # Créer une vidéo finale
        self._create_final_video()
    
    def _log_progress(self, episode, total_episodes, start_time):
        """Affiche les progrès d'entraînement"""
        recent_rewards = self.episode_rewards[-self.config['log_interval']:]
        recent_lengths = self.episode_lengths[-self.config['log_interval']:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        
        # Taux de succès et de contact
        recent_episodes = min(100, len(self.episode_rewards))
        if recent_episodes > 0:
            # Estimer le succès basé sur les récompenses élevées
            success_threshold = 5.0
            recent_successes = [r > success_threshold for r in self.episode_rewards[-recent_episodes:]]
            success_rate = np.mean(recent_successes) * 100
        else:
            success_rate = 0
        
        # Temps écoulé et estimation
        elapsed_time = time.time() - start_time
        time_per_episode = elapsed_time / episode
        remaining_time = (total_episodes - episode) * time_per_episode
        
        # Métriques d'entraînement
        training_info = ""
        if self.training_metrics:
            recent_metrics = self.training_metrics[-10:]
            avg_actor_loss = np.mean([m['actor_loss'] for m in recent_metrics])
            avg_critic_loss = np.mean([m['critic_loss'] for m in recent_metrics])
            avg_alpha = np.mean([m['alpha'] for m in recent_metrics])
            training_info = f"Actor: {avg_actor_loss:.3f}, Critic: {avg_critic_loss:.3f}, Alpha: {avg_alpha:.3f}"
        
        print(f"\n📊 Épisode {episode}/{total_episodes}")
        print(f"   Récompense: {avg_reward:.2f} ± {np.std(recent_rewards):.2f}")
        print(f"   Longueur: {avg_length:.1f}")
        print(f"   Succès: {success_rate:.1f}%")
        print(f"   Niveau curriculum: {self.env.curriculum_level}")
        print(f"   Buffer: {len(self.agent.replay_buffer)}")
        if training_info:
            print(f"   Losses: {training_info}")
        print(f"   Temps: {elapsed_time/60:.1f}min (reste ~{remaining_time/60:.1f}min)")
    
    def _save_checkpoint(self, episode):
        """Sauvegarde un checkpoint"""
        checkpoint_path = self.output_dir / "models" / f"checkpoint_episode_{episode}.pth"
        self.agent.save(checkpoint_path)
        
        # Sauvegarder aussi les métriques
        metrics_path = self.output_dir / "logs" / f"metrics_episode_{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "episode": episode,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "training_metrics": self.training_metrics[-100:] if self.training_metrics else []
            }, f, indent=2)
    
    def _record_evaluation_video(self, episode):
        """Enregistre une vidéo d'évaluation"""
        print(f"🎬 Enregistrement vidéo épisode {episode}")
        
        # Créer un environnement séparé pour l'évaluation
        eval_env = SimpleGraspEnv(
            xml_path=self.config['model_path'],
            max_episode_steps=self.config['max_episode_steps'],
            curriculum_level=self.env.curriculum_level
        )
        
        try:
            video_path, episode_info = self.video_recorder.record_episode(
                eval_env, 
                self.agent, 
                max_steps=500,
                render_mode="rgb_array"
            )
            
            if video_path:
                print(f"   Vidéo sauvegardée: {Path(video_path).name}")
                print(f"   Récompense: {episode_info['total_reward']:.2f}")
                print(f"   Succès: {episode_info['success']}")
        
        except Exception as e:
            print(f"⚠️  Erreur lors de l'enregistrement vidéo: {e}")
        
        finally:
            eval_env.close()
    
    def _save_final_results(self):
        """Sauvegarde les résultats finaux"""
        # Sauvegarder le modèle final
        final_model_path = self.output_dir / "models" / "final_model.pth"
        self.agent.save(final_model_path)
        
        # Créer les graphiques
        self._create_training_plots()
        
        # Sauvegarder les métriques complètes
        final_metrics = {
            "config": self.config,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_metrics": self.training_metrics,
            "final_stats": {
                "total_episodes": len(self.episode_rewards),
                "avg_reward_last_100": float(np.mean(self.episode_rewards[-100:])),
                "max_reward": float(np.max(self.episode_rewards)),
                "avg_length": float(np.mean(self.episode_lengths)),
                "buffer_size": len(self.agent.replay_buffer)
            }
        }
        
        final_metrics_path = self.output_dir / "logs" / "final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✅ Résultats sauvegardés dans {self.output_dir}")
    
    def _create_training_plots(self):
        """Crée les graphiques d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses par épisode
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].grid(True)
        
        # Moyenne mobile des récompenses
        window = min(100, len(self.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moyenne mobile des récompenses (fenêtre={window})')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Récompense moyenne')
            axes[0, 1].grid(True)
        
        # Longueurs des épisodes
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Longueur des épisodes')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Étapes')
        axes[1, 0].grid(True)
        
        # Pertes d'entraînement
        if self.training_metrics:
            actor_losses = [m['actor_loss'] for m in self.training_metrics]
            critic_losses = [m['critic_loss'] for m in self.training_metrics]
            
            axes[1, 1].plot(actor_losses, label='Actor Loss', alpha=0.7)
            axes[1, 1].plot(critic_losses, label='Critic Loss', alpha=0.7)
            axes[1, 1].set_title('Pertes d\'entraînement')
            axes[1, 1].set_xlabel('Étape d\'entraînement')
            axes[1, 1].set_ylabel('Perte')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / "logs" / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Graphiques sauvegardés: {plot_path}")
    
    def _create_final_video(self):
        """Crée une vidéo finale de démonstration"""
        print("🎬 Création de la vidéo finale...")
        
        try:
            video_path, info = self.video_recorder.create_training_video(
                self.env, 
                self.agent, 
                num_episodes=5
            )
            
            if video_path:
                print(f"✅ Vidéo finale créée: {Path(video_path).name}")
                print(f"   Récompense moyenne: {info['avg_reward']:.2f}")
        
        except Exception as e:
            print(f"⚠️  Erreur lors de la création de la vidéo finale: {e}")

def load_config():
    """Charge la configuration d'entraînement"""
    return {
        # Environnement
        'model_path': '/home/oussema/Documents/project/results/g1_combined.xml',
        'max_episode_steps': 500,
        'curriculum_level': 1,
        
        # Entraînement
        'total_episodes': 2000,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 100000,
        'updates_per_episode': 1,
        'hidden_sizes': [256, 256],
        
        # Curriculum learning
        'curriculum_threshold': 0.7,  # Taux de succès pour passer au niveau suivant
        'episodes_per_level': 200,
        
        # Logging et sauvegarde
        'log_interval': 50,
        'save_interval': 200,
        'video_interval': 100,
        'video_fps': 30,
        
        # Sortie
        'output_dir': '/home/oussema/Documents/project/training_results'
    }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement de saisie G1')
    parser.add_argument('--episodes', type=int, default=2000, help='Nombre d\'épisodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Taux d\'apprentissage')
    parser.add_argument('--output', type=str, default='/home/oussema/Documents/project/training_results', help='Dossier de sortie')
    parser.add_argument('--curriculum', type=int, default=1, help='Niveau de curriculum initial')
    parser.add_argument('--video', action='store_true', help='Enregistrer des vidéos')
    
    args = parser.parse_args()
    
    # Charger et modifier la configuration
    config = load_config()
    config['total_episodes'] = args.episodes
    config['learning_rate'] = args.lr
    config['output_dir'] = args.output
    config['curriculum_level'] = args.curriculum
    
    print("🤖 ENTRAÎNEMENT DE SAISIE G1")
    print("=" * 50)
    print(f"Épisodes: {config['total_episodes']}")
    print(f"Taux d'apprentissage: {config['learning_rate']}")
    print(f"Niveau curriculum: {config['curriculum_level']}")
    print(f"Dossier de sortie: {config['output_dir']}")
    
    # Vérifier que le modèle existe
    if not Path(config['model_path']).exists():
        print(f"❌ Modèle non trouvé: {config['model_path']}")
        print("💡 Créez d'abord le modèle avec: python create_combined_model.py")
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
    
    finally:
        print("\n🏁 Fin du programme")

if __name__ == "__main__":
    main()
