#!/usr/bin/env python3
"""
Script d'entraînement stabilisé pour la saisie G1
Corrige les problèmes de stabilité MuJoCo et d'apprentissage
Auteur: Assistant IA
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
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend non-interactif
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Ajouter les modules locaux au path
sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')
sys.path.append('./utils')

if HAS_TORCH:
    from envs.stable_grasp_env import StableGraspEnv
    from agents.improved_sac_agent import ImprovedSACAgent
from utils.video_recorder import VideoRecorder

class StableGraspTrainer:
    """Entraîneur stabilisé pour la tâche de saisie G1"""
    
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
        
        # Initialiser l'environnement stabilisé
        print("🤖 Initialisation de l'environnement stabilisé...")
        self.env = StableGraspEnv(
            xml_path=config['model_path'],
            max_episode_steps=config['max_episode_steps'],
            curriculum_level=config['curriculum_level']
        )
        
        # Initialiser l'agent SAC avec paramètres conservateurs
        print("🧠 Initialisation de l'agent SAC...")
        self.agent = ImprovedSACAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            lr=config['learning_rate'],
            hidden_sizes=config['hidden_sizes'],
            buffer_size=config['buffer_size'],
            gamma=config['gamma'],
            tau=config['tau']
        )
        
        # Enregistreur vidéo avec FFmpeg
        self.video_recorder = VideoRecorder(
            output_dir=self.output_dir / "videos",
            fps=config['video_fps']
        )
        
        # Métriques d'entraînement
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_contacts = []
        self.training_metrics = []
        
        # Variables de stabilité
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        print("✅ Entraîneur stabilisé initialisé")
    
    def train(self):
        """Lance l'entraînement stabilisé"""
        if not HAS_TORCH:
            print("❌ Impossible de lancer l'entraînement sans PyTorch")
            return
            
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT STABILISÉ")
        print("=" * 60)
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        for episode in range(total_episodes):
            episode_start_time = time.time()
            
            # Reset avec gestion d'erreur
            try:
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_contact = False
                
                # Épisode avec protection contre l'instabilité
                done = False
                instability_detected = False
                
                while not done and episode_length < self.config['max_episode_steps']:
                    # Sélection d'action avec exploration réduite au début
                    if episode < 100:  # Phase d'initialisation conservative
                        action = self.agent.select_action(obs, evaluate=True)
                        action = action * 0.1  # Actions très petites
                    else:
                        action = self.agent.select_action(obs)
                    
                    # Étape d'environnement avec gestion d'erreur
                    try:
                        next_obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        
                        # Vérifier si l'épisode a échoué à cause d'instabilité
                        if "error" in info:
                            instability_detected = True
                            reward = -10.0
                            done = True
                            print(f"⚠️  Instabilité détectée épisode {episode}: {info['error']}")
                        
                        # Stocker la transition si pas d'instabilité
                        if not instability_detected:
                            self.agent.store_transition(obs, action, reward, next_obs, done)
                        
                        # Mise à jour des métriques
                        episode_reward += reward
                        episode_length += 1
                        episode_contact = episode_contact or info.get('contact', False)
                        episode_success = terminated and info.get('cube_height', 0) > 0.1
                        
                        obs = next_obs
                        
                    except Exception as e:
                        print(f"⚠️  Erreur durant l'épisode {episode}: {e}")
                        instability_detected = True
                        reward = -10.0
                        done = True
                        break
                
                # Gestion des échecs consécutifs
                if instability_detected:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        print("🛑 Trop d'échecs consécutifs, arrêt de l'entraînement")
                        break
                else:
                    self.consecutive_failures = 0
                
                # Entraînement de l'agent (moins fréquent au début)
                if (len(self.agent.replay_buffer) > self.config['batch_size'] and 
                    not instability_detected and 
                    episode % self.config['training_frequency'] == 0):
                    
                    for _ in range(self.config['updates_per_episode']):
                        training_info = self.agent.update(self.config['batch_size'])
                        if training_info:
                            self.training_metrics.append(training_info)
                
                # Enregistrer les métriques
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_successes.append(episode_success)
                self.episode_contacts.append(episode_contact)
                
                # Affichage des progrès
                if (episode + 1) % self.config['log_interval'] == 0:
                    self._log_progress(episode + 1, total_episodes, start_time)
                
                # Sauvegarde périodique
                if (episode + 1) % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode + 1)
                
                # Enregistrement vidéo périodique (moins fréquent)
                if (episode + 1) % self.config['video_interval'] == 0 and not instability_detected:
                    try:
                        self._record_evaluation_video(episode + 1)
                    except Exception as e:
                        print(f"⚠️  Erreur vidéo: {e}")
                
            except Exception as e:
                print(f"❌ Erreur critique épisode {episode}: {e}")
                continue
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ")
        print(f"   Durée totale: {total_time/3600:.1f}h")
        print(f"   Épisodes: {len(self.episode_rewards)}")
        
        if self.episode_rewards:
            print(f"   Récompense moyenne finale: {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"   Taux de succès: {np.mean(self.episode_successes[-100:]) * 100:.1f}%")
            print(f"   Taux de contact: {np.mean(self.episode_contacts[-100:]) * 100:.1f}%")
        
        # Sauvegarde finale
        self._save_final_results()
        
        # Créer une vidéo finale si possible
        try:
            self._create_final_video()
        except Exception as e:
            print(f"⚠️  Impossible de créer la vidéo finale: {e}")
    
    def _log_progress(self, episode, total_episodes, start_time):
        """Affiche les progrès d'entraînement"""
        recent_episodes = min(self.config['log_interval'], len(self.episode_rewards))
        
        if recent_episodes > 0:
            recent_rewards = self.episode_rewards[-recent_episodes:]
            recent_lengths = self.episode_lengths[-recent_episodes:]
            recent_successes = self.episode_successes[-recent_episodes:]
            recent_contacts = self.episode_contacts[-recent_episodes:]
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes) * 100
            contact_rate = np.mean(recent_contacts) * 100
            
            elapsed_time = time.time() - start_time
            time_per_episode = elapsed_time / episode
            remaining_time = (total_episodes - episode) * time_per_episode
            
            print(f"\n📊 Épisode {episode}/{total_episodes}")
            print(f"   Récompense: {avg_reward:.2f} ± {np.std(recent_rewards):.2f}")
            print(f"   Longueur: {avg_length:.1f}")
            print(f"   Succès: {success_rate:.1f}%")
            print(f"   Contact: {contact_rate:.1f}%")
            print(f"   Échecs consécutifs: {self.consecutive_failures}")
            print(f"   Buffer: {len(self.agent.replay_buffer)}")
            print(f"   Temps: {elapsed_time/60:.1f}min (reste ~{remaining_time/60:.1f}min)")
            
            # Métriques d'entraînement
            if self.training_metrics:
                recent_metrics = self.training_metrics[-10:]
                if recent_metrics:
                    avg_actor_loss = np.mean([m['actor_loss'] for m in recent_metrics])
                    avg_critic_loss = np.mean([m['critic_loss'] for m in recent_metrics])
                    print(f"   Losses: Actor={avg_actor_loss:.3f}, Critic={avg_critic_loss:.3f}")
    
    def _save_checkpoint(self, episode):
        """Sauvegarde un checkpoint"""
        try:
            checkpoint_path = self.output_dir / "models" / f"checkpoint_episode_{episode}.pth"
            self.agent.save(checkpoint_path)
            
            # Sauvegarder aussi les métriques
            metrics_path = self.output_dir / "logs" / f"metrics_episode_{episode}.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "episode": episode,
                    "episode_rewards": self.episode_rewards,
                    "episode_lengths": self.episode_lengths,
                    "episode_successes": self.episode_successes,
                    "episode_contacts": self.episode_contacts,
                    "training_metrics": self.training_metrics[-100:] if self.training_metrics else []
                }, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde checkpoint: {e}")
    
    def _record_evaluation_video(self, episode):
        """Enregistre une vidéo d'évaluation"""
        print(f"🎬 Enregistrement vidéo épisode {episode}")
        
        try:
            # Créer un environnement séparé pour l'évaluation
            eval_env = StableGraspEnv(
                xml_path=self.config['model_path'],
                max_episode_steps=100,  # Plus court pour l'évaluation
                curriculum_level=self.env.curriculum_level
            )
            
            video_path, episode_info = self.video_recorder.record_episode(
                eval_env, 
                self.agent, 
                max_steps=100,
                render_mode="rgb_array"
            )
            
            if video_path:
                print(f"   Vidéo sauvegardée: {Path(video_path).name}")
                print(f"   Récompense: {episode_info['total_reward']:.2f}")
                print(f"   Succès: {episode_info['success']}")
        
        except Exception as e:
            print(f"⚠️  Erreur lors de l'enregistrement vidéo: {e}")
        
        finally:
            try:
                eval_env.close()
            except:
                pass
    
    def _save_final_results(self):
        """Sauvegarde les résultats finaux"""
        try:
            # Sauvegarder le modèle final
            final_model_path = self.output_dir / "models" / "final_model.pth"
            self.agent.save(final_model_path)
            
            # Créer les graphiques si matplotlib disponible
            if HAS_MATPLOTLIB:
                self._create_training_plots()
            
            # Sauvegarder les métriques complètes
            final_metrics = {
                "config": self.config,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "episode_contacts": self.episode_contacts,
                "training_metrics": self.training_metrics,
                "final_stats": {
                    "total_episodes": len(self.episode_rewards),
                    "avg_reward_last_100": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
                    "max_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
                    "success_rate_last_100": float(np.mean(self.episode_successes[-100:])) if self.episode_successes else 0,
                    "contact_rate_last_100": float(np.mean(self.episode_contacts[-100:])) if self.episode_contacts else 0,
                    "buffer_size": len(self.agent.replay_buffer)
                }
            }
            
            final_metrics_path = self.output_dir / "logs" / "final_metrics.json"
            with open(final_metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            print(f"✅ Résultats sauvegardés dans {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde finale: {e}")
    
    def _create_training_plots(self):
        """Crée les graphiques d'entraînement"""
        try:
            if not self.episode_rewards:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Récompenses par épisode
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Récompenses par épisode')
            axes[0, 0].set_xlabel('Épisode')
            axes[0, 0].set_ylabel('Récompense')
            axes[0, 0].grid(True)
            
            # Moyenne mobile des récompenses
            window = min(50, len(self.episode_rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(moving_avg)
                axes[0, 1].set_title(f'Moyenne mobile récompenses (fenêtre={window})')
                axes[0, 1].set_xlabel('Épisode')
                axes[0, 1].set_ylabel('Récompense moyenne')
                axes[0, 1].grid(True)
            
            # Taux de succès
            window = min(50, len(self.episode_successes))
            if window > 1:
                success_moving_avg = np.convolve(self.episode_successes, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(success_moving_avg)
                axes[1, 0].set_title(f'Taux de succès (fenêtre={window})')
                axes[1, 0].set_xlabel('Épisode')
                axes[1, 0].set_ylabel('Taux de succès')
                axes[1, 0].grid(True)
            
            # Longueurs des épisodes
            axes[1, 1].plot(self.episode_lengths)
            axes[1, 1].set_title('Longueur des épisodes')
            axes[1, 1].set_xlabel('Épisode')
            axes[1, 1].set_ylabel('Étapes')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = self.output_dir / "logs" / "training_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Graphiques sauvegardés: {plot_path}")
            
        except Exception as e:
            print(f"⚠️  Erreur création graphiques: {e}")
    
    def _create_final_video(self):
        """Crée une vidéo finale de démonstration"""
        print("🎬 Tentative de création vidéo finale...")
        
        # Pas de vidéo finale pour éviter les erreurs FFmpeg
        print("⚠️  Vidéo finale désactivée pour éviter les erreurs FFmpeg")

def load_stable_config():
    """Charge la configuration d'entraînement stabilisée"""
    return {
        # Environnement
        'model_path': 'results/g1_combined.xml',
        'max_episode_steps': 200,  # Plus court pour stabilité
        'curriculum_level': 1,
        
        # Entraînement - paramètres conservateurs
        'total_episodes': 500,  # Moins d'épisodes pour commencer
        'learning_rate': 1e-4,  # Learning rate plus bas
        'batch_size': 64,       # Batch size plus petit
        'buffer_size': 10000,   # Buffer plus petit
        'updates_per_episode': 1,
        'training_frequency': 5,  # Entraîner moins souvent
        'hidden_sizes': [128, 128],  # Réseaux plus petits
        'gamma': 0.95,          # Discount factor plus conservateur
        'tau': 0.01,            # Soft update plus lent
        
        # Logging et sauvegarde
        'log_interval': 25,     # Logger plus souvent
        'save_interval': 100,
        'video_interval': 200,  # Vidéos moins fréquentes
        'video_fps': 15,        # FPS plus bas
        
        # Sortie
        'output_dir': 'training_results'
    }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement stabilisé de saisie G1')
    parser.add_argument('--episodes', type=int, default=500, help='Nombre d\'épisodes')
    parser.add_argument('--lr', type=float, default=1e-4, help='Taux d\'apprentissage')
    parser.add_argument('--output', type=str, default='training_results', help='Dossier de sortie')
    parser.add_argument('--stable', action='store_true', help='Mode ultra-stable', default=True)
    
    args = parser.parse_args()
    
    # Charger et modifier la configuration
    config = load_stable_config()
    config['total_episodes'] = args.episodes
    config['learning_rate'] = args.lr
    config['output_dir'] = args.output
    
    if args.stable:
        # Mode ultra-stable
        config['learning_rate'] *= 0.5
        config['batch_size'] = 32
        config['training_frequency'] = 10
        print("🛡️  Mode ultra-stable activé")
    
    print("🤖 ENTRAÎNEMENT STABILISÉ DE SAISIE G1")
    print("=" * 50)
    print(f"Épisodes: {config['total_episodes']}")
    print(f"Taux d'apprentissage: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Fréquence d'entraînement: {config['training_frequency']}")
    print(f"Dossier de sortie: {config['output_dir']}")
    
    # Vérifier que le modèle existe
    if not Path(config['model_path']).exists():
        print(f"❌ Modèle non trouvé: {config['model_path']}")
        print("💡 Placez le modèle g1_combined.xml dans le dossier results/")
        return
    
    # Installer FFmpeg si nécessaire pour les vidéos
    print("\n💡 Pour les vidéos, installez FFmpeg:")
    print("   pip install imageio[ffmpeg]")
    print("   ou: sudo apt install ffmpeg")
    
    # Créer et lancer l'entraîneur
    try:
        trainer = StableGraspTrainer(config)
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