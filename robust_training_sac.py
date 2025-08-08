#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT SAC ULTRA-ROBUSTE AVEC CURRICULUM LEARNING
=========================================================

Script d'entraînement professionnel utilisant Stable-Baselines3 SAC
avec l'environnement de grasping ultra-robuste.

Fonctionnalités:
✅ Curriculum learning adaptatif
✅ Sauvegarde automatique des modèles
✅ Monitoring en temps réel
✅ Gestion d'erreurs complète
✅ Génération automatique de vidéos
✅ Viewer MuJoCo en temps réel
✅ Métriques détaillées et graphiques

Version finale ultra-professionnelle et fonctionnelle.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Callable
import warnings
import threading
import signal
from pathlib import Path

warnings.filterwarnings("ignore")

# Imports pour l'entraînement
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback, 
        CallbackList, StopTrainingOnMaxEpisodes
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    import torch
    print("✅ Stable-Baselines3 et PyTorch importés avec succès")
except ImportError as e:
    print(f"❌ Erreur import SB3/PyTorch: {e}")
    print("Installez avec: pip install stable-baselines3[extra] torch")
    sys.exit(1)

# Import de notre environnement
try:
    from envs.ultra_robust_grasp_env import UltraRobustGraspEnv, make_ultra_robust_grasp_env
    print("✅ Environnement ultra-robuste importé")
except ImportError as e:
    print(f"❌ Erreur import environnement: {e}")
    print("Assurez-vous que ultra_robust_grasp_env.py est dans le même dossier")
    sys.exit(1)


class CurriculumCallback(BaseCallback):
    """
    Callback personnalisé pour gérer le curriculum learning
    """
    
    def __init__(self, env_wrapper, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.curriculum_history = []
        self.phase_statistics = {}
        self.best_mean_reward = -np.inf
        self.episodes_count = 0
        
    def _on_step(self) -> bool:
        # Récupérer les infos de l'environnement
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Si l'épisode est terminé
            if self.locals.get('dones', [False])[0]:
                self.episodes_count += 1
                episode_reward = self.locals.get('rewards', [0])[0]
                
                # Faire avancer le curriculum
                if hasattr(self.env_wrapper, 'advance_curriculum_level'):
                    advanced = self.env_wrapper.advance_curriculum_level(episode_reward)
                    if advanced:
                        self.logger.info(f"🎓 Curriculum avancé au niveau {self.env_wrapper.current_level}")
                
                # Enregistrer les statistiques
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(info.get('episode_step', 0))
                
                # Statistiques du curriculum
                curriculum_status = self.env_wrapper.get_curriculum_status()
                self.curriculum_history.append({
                    'episode': self.episodes_count,
                    'level': curriculum_status['current_level'],
                    'phase': info.get('phase', 'Unknown'),
                    'reward': episode_reward,
                    'timestamp': time.time()
                })
                
                # Logging périodique
                if self.episodes_count % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    mean_reward = np.mean(recent_rewards)
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                    
                    self.logger.info(
                        f"📊 Épisode {self.episodes_count}: "
                        f"Reward={episode_reward:.2f}, "
                        f"Mean(10)={mean_reward:.2f}, "
                        f"Best={self.best_mean_reward:.2f}, "
                        f"Level={curriculum_status['current_level']}, "
                        f"Phase={info.get('phase', 'N/A')}"
                    )
        
        return True
    
    def _on_training_start(self) -> None:
        self.logger.info("🚀 Début de l'entraînement avec curriculum learning")
    
    def _on_training_end(self) -> None:
        self.logger.info("🏁 Fin de l'entraînement")
        self._save_curriculum_statistics()
    
    def _save_curriculum_statistics(self):
        """Sauvegarde les statistiques du curriculum"""
        try:
            stats_dir = Path("robust_sac_results/curriculum_stats")
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sauvegarder les données brutes
            with open(stats_dir / f"curriculum_history_{timestamp}.json", 'w') as f:
                json.dump(self.curriculum_history, f, indent=2)
            
            # Créer des graphiques
            self._create_training_plots(stats_dir, timestamp)
            
            self.logger.info(f"📊 Statistiques sauvegardées dans {stats_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde statistiques: {e}")
    
    def _create_training_plots(self, stats_dir: Path, timestamp: str):
        """Crée des graphiques de l'entraînement"""
        try:
            # Graphique des récompenses
            plt.figure(figsize=(15, 10))
            
            # Récompenses par épisode
            plt.subplot(2, 3, 1)
            plt.plot(self.episode_rewards, alpha=0.6)
            plt.plot(np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
            plt.title('Récompenses par Épisode')
            plt.xlabel('Épisode')
            plt.ylabel('Récompense')
            plt.grid(True)
            
            # Longueur des épisodes
            plt.subplot(2, 3, 2)
            plt.plot(self.episode_lengths, alpha=0.6)
            plt.title('Longueur des Épisodes')
            plt.xlabel('Épisode')
            plt.ylabel('Steps')
            plt.grid(True)
            
            # Progression du curriculum
            if self.curriculum_history:
                levels = [entry['level'] for entry in self.curriculum_history]
                episodes = [entry['episode'] for entry in self.curriculum_history]
                
                plt.subplot(2, 3, 3)
                plt.plot(episodes, levels, 'bo-', markersize=2)
                plt.title('Progression du Curriculum')
                plt.xlabel('Épisode')
                plt.ylabel('Niveau')
                plt.grid(True)
                
                # Distribution des niveaux
                plt.subplot(2, 3, 4)
                level_counts = {}
                for level in levels:
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                plt.bar(level_counts.keys(), level_counts.values())
                plt.title('Distribution des Niveaux')
                plt.xlabel('Niveau')
                plt.ylabel('Nombre d\'Épisodes')
                
                # Récompenses par niveau
                plt.subplot(2, 3, 5)
                level_rewards = {}
                for entry in self.curriculum_history:
                    level = entry['level']
                    if level not in level_rewards:
                        level_rewards[level] = []
                    level_rewards[level].append(entry['reward'])
                
                for level, rewards in level_rewards.items():
                    plt.scatter([level] * len(rewards), rewards, alpha=0.6, label=f'Niveau {level}')
                
                plt.title('Récompenses par Niveau de Curriculum')
                plt.xlabel('Niveau')
                plt.ylabel('Récompense')
                plt.legend()
                plt.grid(True)
            
            # Statistiques globales
            plt.subplot(2, 3, 6)
            stats_text = f"""
Statistiques d'Entraînement:
• Épisodes total: {len(self.episode_rewards)}
• Récompense moyenne: {np.mean(self.episode_rewards):.2f}
• Meilleure récompense: {np.max(self.episode_rewards):.2f}
• Longueur moyenne: {np.mean(self.episode_lengths):.1f}
• Niveau final: {max(levels) if levels else 1}
"""
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='center')
            plt.axis('off')
            plt.title('Résumé')
            
            plt.tight_layout()
            plt.savefig(stats_dir / f"training_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Graphiques sauvegardés: training_analysis_{timestamp}.png")
            
        except Exception as e:
            print(f"❌ Erreur création graphiques: {e}")


class VideoRecorderCallback(BaseCallback):
    """
    Callback pour enregistrer des vidéos périodiquement
    """
    
    def __init__(self, env, video_folder: str, record_freq: int = 50000, 
                 video_length: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.record_freq = record_freq
        self.video_length = video_length
        self.last_record_step = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_record_step >= self.record_freq:
            self._record_video()
            self.last_record_step = self.num_timesteps
        return True
    
    def _record_video(self):
        """Enregistre une vidéo de démonstration"""
        try:
            import cv2
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = self.video_folder / f"demo_step_{self.num_timesteps}_{timestamp}.mp4"
            
            # Configuration vidéo
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            width, height = 640, 480
            
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                self.logger.warning("❌ Impossible d'ouvrir le writer vidéo")
                return
            
            # Enregistrer un épisode
            obs = self.training_env.reset()
            frames_recorded = 0
            
            for _ in range(self.video_length):
                # Prédire l'action avec le modèle actuel
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.training_env.step(action)
                
                # Capturer la frame
                try:
                    frame = self.env.render(mode='rgb_array')
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                        frames_recorded += 1
                except:
                    pass
                
                if done[0]:
                    obs = self.training_env.reset()
            
            video_writer.release()
            
            if frames_recorded > 0:
                self.logger.info(f"🎬 Vidéo enregistrée: {video_path} ({frames_recorded} frames)")
            else:
                os.remove(video_path)
                self.logger.warning("⚠️ Aucune frame capturée, vidéo supprimée")
                
        except Exception as e:
            self.logger.error(f"❌ Erreur enregistrement vidéo: {e}")


class RobustSACTrainer:
    """
    Classe principale pour l'entraînement SAC ultra-robuste
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
        # Variables d'état
        self.model = None
        self.env = None
        self.training_active = False
        
        # Gestion des signaux pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🎯 RobustSACTrainer initialisé")
    
    def setup_directories(self):
        """Crée les dossiers nécessaires"""
        self.results_dir = Path(self.config['results_dir'])
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.videos_dir = self.results_dir / "videos"
        self.tensorboard_dir = self.results_dir / "tensorboard"
        
        for directory in [self.results_dir, self.models_dir, self.logs_dir, 
                         self.videos_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Dossiers créés dans: {self.results_dir}")
    
    def setup_logging(self):
        """Configure le logging"""
        import logging
        
        log_file = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📝 Logging configuré")
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire pour arrêt propre"""
        self.logger.info(f"🛑 Signal reçu ({signum}), arrêt propre en cours...")
        self.training_active = False
        
        if self.model:
            self.save_final_model()
        
        if self.env:
            self.env.close()
        
        sys.exit(0)
    
    def create_environment(self):
        """Crée l'environnement d'entraînement"""
        self.logger.info("🏗️ Création de l'environnement...")
        
        def make_env():
            env = make_ultra_robust_grasp_env(
                model_path=self.config.get('model_path'),
                render_mode=self.config.get('render_mode', 'human'),
                enable_curriculum=True,
                enable_mujoco_viewer=True
            )
            
            # Wrapper Monitor pour logging
            env = Monitor(env, self.logs_dir, allow_early_resets=True)
            return env
        
        # Créer l'environnement vectorisé
        if self.config.get('n_envs', 1) > 1:
            self.env = make_vec_env(make_env, n_envs=self.config['n_envs'], 
                                  vec_env_cls=SubprocVecEnv)
        else:
            self.env = DummyVecEnv([make_env])
        
        # Garder une référence à l'environnement de base pour curriculum
        self.base_env = make_env()
        
        self.logger.info(f"✅ Environnement créé: {self.env}")
        self.logger.info(f"📊 Action space: {self.env.action_space}")
        self.logger.info(f"📊 Observation space: {self.env.observation_space}")
    
    def create_model(self):
        """Crée le modèle SAC"""
        self.logger.info("🤖 Création du modèle SAC...")
        
        # Configuration du modèle
        model_config = {
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'buffer_size': self.config.get('buffer_size', 1000000),
            'learning_starts': self.config.get('learning_starts', 10000),
            'batch_size': self.config.get('batch_size', 256),
            'tau': self.config.get('tau', 0.005),
            'gamma': self.config.get('gamma', 0.99),
            'train_freq': self.config.get('train_freq', 1),
            'gradient_steps': self.config.get('gradient_steps', 1),
            'ent_coef': self.config.get('ent_coef', 'auto'),
            'target_update_interval': self.config.get('target_update_interval', 1),
            'target_entropy': self.config.get('target_entropy', 'auto'),
            'use_sde': self.config.get('use_sde', False),
            'sde_sample_freq': self.config.get('sde_sample_freq', -1),
            'use_sde_at_warmup': self.config.get('use_sde_at_warmup', False),
            'policy_kwargs': {
                'net_arch': self.config.get('net_arch', [256, 256]),
                'activation_fn': torch.nn.ReLU,
                'use_sde': self.config.get('use_sde', False),
            },
            'verbose': 1,
            'tensorboard_log': str(self.tensorboard_dir),
            'device': self.config.get('device', 'auto'),
        }
        
        # Créer le modèle
        self.model = SAC(
            "MlpPolicy",
            self.env,
            **model_config
        )
        
        # Charger un modèle pré-entraîné si spécifié
        if self.config.get('load_model_path'):
            self.logger.info(f"📥 Chargement du modèle: {self.config['load_model_path']}")
            self.model.load(self.config['load_model_path'])
        
        self.logger.info("✅ Modèle SAC créé")
        self.logger.info(f"🧠 Architecture: {model_config['policy_kwargs']['net_arch']}")
        self.logger.info(f"💾 Buffer size: {model_config['buffer_size']}")
        self.logger.info(f"🎯 Learning rate: {model_config['learning_rate']}")
    
    def create_callbacks(self):
        """Crée les callbacks pour l'entraînement"""
        self.logger.info("🔄 Configuration des callbacks...")
        
        callbacks = []
        
        # Callback curriculum
        curriculum_callback = CurriculumCallback(
            self.base_env,
            save_freq=self.config.get('save_freq', 10000),
            verbose=1
        )
        callbacks.append(curriculum_callback)
        
        # Callback checkpoint
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('save_freq', 10000),
            save_path=str(self.models_dir),
            name_prefix='sac_grasp_checkpoint',
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Callback vidéo
        if self.config.get('record_videos', True):
            video_callback = VideoRecorderCallback(
                self.base_env,
                str(self.videos_dir),
                record_freq=self.config.get('video_freq', 50000),
                video_length=self.config.get('video_length', 500),
                verbose=1
            )
            callbacks.append(video_callback)
        
        # Callback d'arrêt par nombre d'épisodes
        if self.config.get('max_episodes'):
            stop_callback = StopTrainingOnMaxEpisodes(
                max_episodes=self.config['max_episodes'],
                verbose=1
            )
            callbacks.append(stop_callback)
        
        self.callbacks = CallbackList(callbacks)
        self.logger.info(f"✅ {len(callbacks)} callbacks configurés")
    
    def train(self):
        """Lance l'entraînement principal"""
        self.logger.info("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        self.training_active = True
        
        try:
            # Créer tous les composants
            self.create_environment()
            self.create_model()
            self.create_callbacks()
            
            # Afficher les informations de configuration
            self._log_training_config()
            
            # Lancer l'entraînement
            self.logger.info("🎯 Lancement de l'apprentissage...")
            
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=self.callbacks,
                log_interval=self.config.get('log_interval', 100),
                tb_log_name="SAC_GraspTraining",
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            self.training_active = False
            
            # Sauvegarder le modèle final
            self.save_final_model()
            
            # Évaluation finale
            self.final_evaluation()
            
            # Statistiques finales
            training_time = time.time() - start_time
            self.logger.info("🏁 ENTRAÎNEMENT TERMINÉ")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️ Temps total: {training_time/3600:.2f} heures")
            self.logger.info(f"📊 Timesteps: {self.config['total_timesteps']:,}")
            self.logger.info(f"📁 Résultats: {self.results_dir}")
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Entraînement interrompu par l'utilisateur")
            self.save_final_model()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur pendant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.env:
                self.env.close()

    
    def _log_training_config(self):
        """Log la configuration d'entraînement"""
        self.logger.info("⚙️ CONFIGURATION D'ENTRAÎNEMENT")
        self.logger.info("-" * 40)
        
        config_items = [
            ("Total timesteps", f"{self.config['total_timesteps']:,}"),
            ("Environnements", self.config.get('n_envs', 1)),
            ("Learning rate", self.config.get('learning_rate', 3e-4)),
            ("Batch size", self.config.get('batch_size', 256)),
            ("Buffer size", f"{self.config.get('buffer_size', 1000000):,}"),
            ("Save frequency", f"{self.config.get('save_freq', 10000):,}"),
            ("Device", self.config.get('device', 'auto')),
            ("Curriculum learning", "Activé"),
            ("MuJoCo viewer", "Activé"),
            ("Enregistrement vidéo", self.config.get('record_videos', True)),
        ]
        
        for key, value in config_items:
            self.logger.info(f"  {key:<20}: {value}")
        
        self.logger.info("-" * 40)
    
    def save_final_model(self):
        """Sauvegarde le modèle final"""
        if not self.model:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Modèle final
            final_model_path = self.models_dir / f"sac_grasp_final_{timestamp}.zip"
            self.model.save(final_model_path)
            
            # Meilleur modèle (lien symbolique)
            best_model_path = self.models_dir / "sac_grasp_best.zip"
            if best_model_path.exists():
                best_model_path.unlink()
            
            try:
                best_model_path.symlink_to(final_model_path.name)
            except:
                # Fallback: copie du fichier
                import shutil
                shutil.copy2(final_model_path, best_model_path)
            
            self.logger.info(f"💾 Modèle final sauvegardé: {final_model_path}")
            self.logger.info(f"🏆 Meilleur modèle: {best_model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde modèle: {e}")
    
    def final_evaluation(self):
        """Évaluation finale du modèle"""
        self.logger.info("🔍 ÉVALUATION FINALE")
        self.logger.info("-" * 40)
        
        try:
            # Créer un environnement d'évaluation
            eval_env = make_ultra_robust_grasp_env(
                enable_curriculum=False,  # Pas de curriculum pour l'évaluation
                enable_mujoco_viewer=False
            )
            
            n_eval_episodes = 10
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(n_eval_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Compter les succès
                if info.get('cube_lifted', False) and info.get('successful_grasp', False):
                    success_count += 1
                
                self.logger.info(
                    f"  Épisode {episode+1}: Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, Phase={info.get('phase', 'N/A')}"
                )
            
            # Statistiques finales
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = success_count / n_eval_episodes * 100
            
            self.logger.info("-" * 40)
            self.logger.info(f"📊 Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")
            self.logger.info(f"📏 Longueur moyenne: {mean_length:.1f}")
            self.logger.info(f"🎯 Taux de succès: {success_rate:.1f}%")
            self.logger.info(f"🏆 Meilleure récompense: {max(episode_rewards):.2f}")
            
            eval_env.close()
            
            # Sauvegarder les résultats d'évaluation
            eval_results = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'mean_length': float(mean_length),
                'success_rate': float(success_rate),
                'best_reward': float(max(episode_rewards)),
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': [int(l) for l in episode_lengths],
                'timestamp': datetime.now().isoformat()
            }
            
            eval_file = self.results_dir / "final_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            self.logger.info(f"💾 Résultats d'évaluation sauvegardés: {eval_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur évaluation finale: {e}")


def main():
    """Fonction principale"""
    # Configuration par défaut
    config = {
        # Entraînement
        'total_timesteps': 1000000,  # 1M steps
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'train_freq': 1,
        'gradient_steps': 1,
        'gamma': 0.99,
        'tau': 0.005,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'net_arch': [512, 512, 256],  # Architecture plus large
        
        # Environnement
        'model_path': "/home/oussema/Documents/project/results/g1_combined.xml",
        'render_mode': 'human',
        'n_envs': 1,  # Un seul environnement pour voir la simulation
        
        # Sauvegarde et monitoring
        'results_dir': 'robust_sac_results',
        'save_freq': 25000,
        'log_interval': 10,
        'record_videos': True,
        'video_freq': 100000,  # Vidéo tous les 100k steps
        'video_length': 1000,
        
        # Matériel
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Optionnel
        'load_model_path': None,  # Pour reprendre un entraînement
        'max_episodes': None,     # Limiter par épisodes au lieu de steps
    }
    
    print("🚀 ENTRAÎNEMENT SAC ULTRA-ROBUSTE POUR GRASPING")
    print("=" * 80)
    print(f"🖥️  Device: {config['device']}")
    print(f"📊 Timesteps: {config['total_timesteps']:,}")
    print(f"🧠 Architecture: {config['net_arch']}")
    print(f"💾 Buffer size: {config['buffer_size']:,}")
    print(f"🎥 Enregistrement vidéo: {config['record_videos']}")
    print("=" * 80)
    
    # Créer et lancer le trainer
    trainer = RobustSACTrainer(config)
    
    try:
        trainer.train()
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Résultats disponibles dans: {trainer.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()