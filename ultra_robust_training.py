#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT ULTRA-ROBUSTE - VERSION FINALE CORRIGÉE
=======================================================

Script d'entraînement qui implémente TOUS les insights du collègue:
✅ Utilise l'environnement ultra-robuste 
✅ TD3 avec hyperparamètres optimisés
✅ Callbacks sécurisés avec gestion d'erreurs CORRIGÉE
✅ Pas de double création d'environnements
✅ Évaluation et vidéo intégrées
✅ Monitoring avancé des performances
✅ Problème logger résolu
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import signal
import logging

# Configuration des warnings
warnings.filterwarnings("ignore")

# Imports ML
try:
    import torch
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure
    print("✅ Stable-Baselines3 et PyTorch importés")
except ImportError as e:
    print(f"❌ Erreur import ML: {e}")
    sys.exit(1)

# Imports vidéo
try:
    import imageio
    from PIL import Image
    print("✅ Outils vidéo importés")
except ImportError:
    print("⚠️ Outils vidéo non disponibles")
    imageio = None

# Import de notre environnement ultra-robuste
try:
    from envs.ultra_robust_grasp_env import UltraRobustGraspEnv, make_ultra_robust_grasp_env
    print("✅ Environnement ultra-robuste importé")
except ImportError:
    try:
        # Si le fichier est dans le dossier courant
        from envs.ultra_robust_grasp_env import UltraRobustGraspEnv, make_ultra_robust_grasp_env
        print("✅ Environnement ultra-robuste importé (dossier courant)")
    except ImportError:
        print("❌ ERREUR: Impossible d'importer l'environnement ultra-robuste")
        print("Assurez-vous que le fichier ultra_robust_grasp_env.py est accessible")
        sys.exit(1)


class UltraRobustCallback(BaseCallback):
    """
    Callback ultra-robuste avec toutes les fonctionnalités - LOGGER CORRIGÉ
    """
    
    def __init__(self, 
                 log_freq: int = 1000,
                 eval_freq: int = 25000, 
                 video_freq: int = 50000,
                 save_freq: int = 50000,
                 n_eval_episodes: int = 5,
                 results_dir: str = "ultra_robust_results",
                 verbose: int = 1):
        
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.video_freq = video_freq
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.results_dir = Path(results_dir)
        
        # Créer les dossiers
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "videos").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        
        # Métriques de suivi
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.distances = []
        self.contact_rates = []
        self.best_eval_reward = -np.inf
        
        # Environnement d'évaluation (créé une seule fois)
        self.eval_env = None
        
        # Logger CORRIGÉ
        self.setup_logging()
    
    def setup_logging(self):
        """Configure le logging - CORRIGÉ"""
        # Utiliser un nom différent pour éviter le conflit avec BaseCallback.logger
        self.custom_logger = logging.getLogger("UltraRobustCallback")
        self.custom_logger.setLevel(logging.INFO)
        
        if not self.custom_logger.handlers:
            # Handler fichier
            handler = logging.FileHandler(self.results_dir / "training.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.custom_logger.addHandler(handler)
            
            # Handler console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.custom_logger.addHandler(console_handler)
    
    def _init_callback(self) -> None:
        """Initialisation du callback"""
        super()._init_callback()
        
        # Créer l'environnement d'évaluation
        try:
            self.eval_env = UltraRobustGraspEnv(render_mode="rgb_array")
            self.custom_logger.info("✅ Environnement d'évaluation créé")
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création env d'évaluation: {e}")
    
    def _on_step(self) -> bool:
        """Appelé à chaque step"""
        
        # Logging périodique
        if self.n_calls % self.log_freq == 0:
            self._log_training_metrics()
        
        # Évaluation périodique
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            self._evaluate_model()
        
        # Vidéo périodique
        if self.n_calls % self.video_freq == 0 and self.n_calls > 0:
            self._create_video()
        
        # Sauvegarde périodique
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_model()
        
        return True
    
    def _log_training_metrics(self):
        """Log des métriques d'entraînement"""
        try:
            # Obtenir les infos de l'environnement
            infos = self.locals.get('infos', [{}])
            if infos and len(infos) > 0:
                info = infos[0]
                
                # Log des métriques de base
                if hasattr(self.logger, 'record'):
                    self.logger.record("train/distance", info.get('distance', 0))
                    self.logger.record("train/contact_count", info.get('contact_count', 0))
                    self.logger.record("train/cube_velocity", info.get('cube_velocity', 0))
                    self.logger.record("train/episode_step", info.get('episode_step', 0))
                
                # Stocker pour les plots
                if 'distance' in info:
                    self.distances.append(info['distance'])
                    if len(self.distances) > 1000:  # Limiter la mémoire
                        self.distances.pop(0)
                
                # Log occasionnel dans la console
                if self.n_calls % (self.log_freq * 10) == 0:
                    self.custom_logger.info(
                        f"Step {self.n_calls}: "
                        f"distance={info.get('distance', 0):.3f}, "
                        f"contacts={info.get('contact_count', 0)}, "
                        f"cube_vel={info.get('cube_velocity', 0):.3f}"
                    )
        
        except Exception as e:
            if self.verbose > 1:
                self.custom_logger.warning(f"Erreur logging: {e}")
    
    def _evaluate_model(self):
        """Évaluation complète du modèle"""
        if not self.eval_env:
            return
        
        self.custom_logger.info(f"📊 Évaluation à {self.n_calls} steps...")
        
        episode_rewards = []
        episode_lengths = []
        final_distances = []
        contact_successes = 0
        
        try:
            for episode in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0.0
                episode_length = 0
                best_contacts = 0
                
                for step in range(500):  # Max 500 steps par épisode
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Suivi des contacts
                    contacts = info.get('contact_count', 0)
                    if contacts > best_contacts:
                        best_contacts = contacts
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                final_distances.append(info.get('distance', float('inf')))
                
                # Succès si au moins 2 contacts établis
                if best_contacts >= 2:
                    contact_successes += 1
        
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur pendant évaluation: {e}")
            return
        
        # Calculer les statistiques
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_distance = np.mean(final_distances)
        success_rate = contact_successes / self.n_eval_episodes * 100
        
        # Stocker les résultats
        self.eval_rewards.append(mean_reward)
        
        # Sauvegarder le meilleur modèle
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            try:
                best_model_path = self.results_dir / "models" / "best_model.zip"
                self.model.save(str(best_model_path))
                self.custom_logger.info(f"💾 Nouveau meilleur modèle sauvegardé: {mean_reward:.2f}")
            except Exception as e:
                self.custom_logger.error(f"❌ Erreur sauvegarde meilleur modèle: {e}")
        
        # Log des résultats
        self.custom_logger.info(f"   Reward moyen: {mean_reward:.2f} ± {std_reward:.2f}")
        self.custom_logger.info(f"   Distance moyenne finale: {mean_distance:.3f}")
        self.custom_logger.info(f"   Taux de succès contacts: {success_rate:.1f}%")
        self.custom_logger.info(f"   Longueur moyenne: {np.mean(episode_lengths):.1f}")
        
        # Log TensorBoard
        if hasattr(self.logger, 'record'):
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_distance", mean_distance)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/mean_length", np.mean(episode_lengths))
    
    def _create_video(self):
        """Crée une vidéo de démonstration"""
        if not self.eval_env or not imageio:
            return
        
        self.custom_logger.info(f"🎥 Création vidéo à {self.n_calls} steps...")
        
        try:
            frames = []
            obs, _ = self.eval_env.reset()
            
            for step in range(300):  # 10 secondes à 30fps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                # Capturer frame
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)
                
                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
            
            # Sauvegarder la vidéo
            if len(frames) > 10:  # Au moins quelques frames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = self.results_dir / "videos" / f"demo_{self.n_calls}_{timestamp}.mp4"
                
                # Convertir en format PIL si nécessaire
                pil_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        pil_frames.append(Image.fromarray(frame.astype(np.uint8)))
                    else:
                        pil_frames.append(frame)
                
                imageio.mimsave(str(video_path), pil_frames, fps=30)
                self.custom_logger.info(f"✅ Vidéo sauvegardée: {video_path.name}")
        
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création vidéo: {e}")
    
    def _save_model(self):
        """Sauvegarde périodique du modèle"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.results_dir / "models" / f"model_{self.n_calls}_{timestamp}.zip"
            self.model.save(str(model_path))
            
            if self.verbose > 0:
                self.custom_logger.info(f"💾 Modèle sauvegardé: {model_path.name}")
        
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur sauvegarde modèle: {e}")
    
    def _on_training_end(self) -> None:
        """Actions à la fin de l'entraînement"""
        self.custom_logger.info("🏁 Fin de l'entraînement - Création des graphiques finaux...")
        
        try:
            self._create_final_plots()
            self._save_training_stats()
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création graphiques finaux: {e}")
        
        # Fermer l'environnement d'évaluation
        if self.eval_env:
            try:
                self.eval_env.close()
            except:
                pass
    
    def _create_final_plots(self):
        """Crée les graphiques finaux de l'entraînement"""
        if not self.eval_rewards and not self.distances:
            self.custom_logger.warning("Pas de données pour les graphiques")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Rewards d'évaluation
            if self.eval_rewards:
                axes[0, 0].plot(self.eval_rewards, 'b-', linewidth=2)
                axes[0, 0].set_title('Rewards d\'Évaluation')
                axes[0, 0].set_xlabel('Évaluations')
                axes[0, 0].set_ylabel('Reward Moyen')
                axes[0, 0].grid(True)
            
            # Distances pendant l'entraînement
            if self.distances:
                # Moyenner sur fenêtres pour lisser
                window_size = min(100, len(self.distances) // 10)
                if window_size > 1:
                    smoothed = np.convolve(self.distances, np.ones(window_size)/window_size, mode='valid')
                    axes[0, 1].plot(smoothed, 'r-', linewidth=2, label='Distance lissée')
                axes[0, 1].plot(self.distances, 'r-', alpha=0.3, label='Distance brute')
                axes[0, 1].set_title('Distance Cube-Palme')
                axes[0, 1].set_xlabel('Steps (x1000)')
                axes[0, 1].set_ylabel('Distance (m)')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Histogramme des distances
            if self.distances:
                axes[1, 0].hist(self.distances, bins=50, alpha=0.7, color='green')
                axes[1, 0].set_title('Distribution des Distances')
                axes[1, 0].set_xlabel('Distance (m)')
                axes[1, 0].set_ylabel('Fréquence')
                axes[1, 0].grid(True)
            
            # Statistiques d'entraînement
            stats_text = f"""Statistiques d'Entraînement:

• Steps total: {self.n_calls:,}
• Évaluations: {len(self.eval_rewards)}
• Meilleur reward: {self.best_eval_reward:.2f}
• Distance moyenne: {np.mean(self.distances):.3f}
• Distance minimale: {np.min(self.distances):.3f}
• Écart-type distance: {np.std(self.distances):.3f}
"""
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Résumé de l\'Entraînement')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.results_dir / "plots" / f"training_summary_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.custom_logger.info(f"📈 Graphiques sauvegardés: {plot_path.name}")
        
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création graphiques: {e}")
    
    def _save_training_stats(self):
        """Sauvegarde les statistiques d'entraînement"""
        try:
            stats = {
                'total_steps': int(self.n_calls),
                'eval_rewards': [float(r) for r in self.eval_rewards],
                'best_eval_reward': float(self.best_eval_reward),
                'distances_sample': [float(d) for d in self.distances[-1000:]] if self.distances else [],
                'mean_distance': float(np.mean(self.distances)) if self.distances else 0,
                'min_distance': float(np.min(self.distances)) if self.distances else 0,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'log_freq': self.log_freq,
                    'eval_freq': self.eval_freq,
                    'video_freq': self.video_freq,
                    'n_eval_episodes': self.n_eval_episodes
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_path = self.results_dir / f"training_stats_{timestamp}.json"
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.custom_logger.info(f"💾 Statistiques sauvegardées: {stats_path.name}")
        
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur sauvegarde statistiques: {e}")


class UltraRobustTrainer:
    """
    Trainer ultra-robuste qui évite tous les problèmes - LOGGER CORRIGÉ
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Dossier de résultats
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Variables d'état
        self.model = None
        self.env = None
        self.training_active = False
        
        # Gestion des signaux pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Logger CORRIGÉ
        self.setup_logging()
        
        self.custom_logger.info("🎯 UltraRobustTrainer initialisé")
    
    def setup_logging(self):
        """Configure le logging global - CORRIGÉ"""
        log_file = self.results_dir / f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Créer le logger personnalisé
        self.custom_logger = logging.getLogger("UltraRobustTrainer")
        self.custom_logger.setLevel(logging.INFO)
        
        # Éviter les doublons de handlers
        if not self.custom_logger.handlers:
            # Handler fichier
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.custom_logger.addHandler(file_handler)
            
            # Handler console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(file_formatter)
            self.custom_logger.addHandler(console_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire d'arrêt propre"""
        self.custom_logger.info(f"🛑 Signal reçu ({signum}), arrêt propre...")
        self.training_active = False
        
        if self.model:
            try:
                emergency_path = self.results_dir / "emergency_save.zip"
                self.model.save(str(emergency_path))
                self.custom_logger.info(f"💾 Sauvegarde d'urgence: {emergency_path}")
            except:
                pass
        
        if self.env:
            try:
                self.env.close()
            except:
                pass
        
        sys.exit(0)
    
    def create_environment(self):
        """Crée l'environnement d'entraînement"""
        self.custom_logger.info("🏗️ Création de l'environnement d'entraînement...")
        
        try:
            # Environnement de base
            base_env = UltraRobustGraspEnv(
                render_mode=self.config.get('render_mode', 'rgb_array'),
                max_episode_steps=self.config.get('max_episode_steps', 500),
                enable_assistance=self.config.get('enable_assistance', True)
            )
            
            # Wrapper Monitor pour logging
            monitor_file = self.results_dir / "monitor.csv"
            env = Monitor(base_env, str(monitor_file))
            
            # Wrapper vectoriel (requis par SB3)
            self.env = DummyVecEnv([lambda: env])
            
            self.custom_logger.info("✅ Environnement créé avec succès")
            self.custom_logger.info(f"   Action space: {self.env.action_space}")
            self.custom_logger.info(f"   Observation space: {self.env.observation_space}")
            
            return self.env
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création environnement: {e}")
            raise
    
    def create_model(self):
        """Crée le modèle TD3 avec configuration robuste"""
        self.custom_logger.info("🧠 Création du modèle TD3...")
        
        try:
            # Configuration des hyperparamètres
            n_actions = self.env.action_space.shape[-1]
            
            # Bruit d'action pour l'exploration
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.config.get('action_noise_sigma', 0.1) * np.ones(n_actions)
            )
            
            # Modèle TD3
            self.model = TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 1000000),
                learning_starts=self.config.get('learning_starts', 25000),
                batch_size=self.config.get('batch_size', 256),
                tau=self.config.get('tau', 0.005),
                gamma=self.config.get('gamma', 0.98),
                train_freq=self.config.get('train_freq', 1),
                gradient_steps=self.config.get('gradient_steps', 1),
                action_noise=action_noise,
                policy_kwargs={
                    'net_arch': self.config.get('net_arch', [400, 300]),
                    'activation_fn': torch.nn.ReLU,
                },
                verbose=1,
                device=self.config.get('device', 'auto'),
                tensorboard_log=str(self.results_dir / "tensorboard")
            )
            
            # Charger un modèle pré-entraîné si spécifié
            if self.config.get('load_model_path'):
                self.custom_logger.info(f"📥 Chargement du modèle: {self.config['load_model_path']}")
                self.model.load(self.config['load_model_path'])
            
            self.custom_logger.info("✅ Modèle TD3 créé avec succès")
            self.custom_logger.info(f"   Device: {self.model.device}")
            self.custom_logger.info(f"   Architecture: {self.config.get('net_arch', [400, 300])}")
            self.custom_logger.info(f"   Buffer size: {self.config.get('buffer_size', 1000000):,}")
            self.custom_logger.info(f"   Learning rate: {self.config.get('learning_rate', 3e-4)}")
            
            return self.model
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création modèle: {e}")
            raise
    
    def create_callbacks(self):
        """Crée les callbacks d'entraînement"""
        self.custom_logger.info("🔄 Configuration des callbacks...")
        
        try:
            callback = UltraRobustCallback(
                log_freq=self.config.get('log_freq', 1000),
                eval_freq=self.config.get('eval_freq', 25000),
                video_freq=self.config.get('video_freq', 50000),
                save_freq=self.config.get('save_freq', 50000),
                n_eval_episodes=self.config.get('n_eval_episodes', 5),
                results_dir=str(self.results_dir),
                verbose=1
            )
            
            self.callbacks = CallbackList([callback])
            self.custom_logger.info("✅ Callbacks configurés")
            
            return self.callbacks
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur configuration callbacks: {e}")
            raise
    
    def train(self):
        """Lance l'entraînement principal"""
        self.custom_logger.info("🚀 DÉBUT DE L'ENTRAÎNEMENT ULTRA-ROBUSTE")
        self.custom_logger.info("=" * 80)
        
        start_time = time.time()
        self.training_active = True
        
        try:
            # Créer tous les composants
            self.create_environment()
            self.create_model() 
            self.create_callbacks()
            
            # Log de la configuration
            self._log_training_config()
            
            # Lancer l'entraînement
            self.custom_logger.info("🎯 Lancement de l'apprentissage TD3...")
            
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=self.callbacks,
                log_interval=self.config.get('log_interval', 10),
                tb_log_name="TD3_UltraRobustGrasp",
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            self.training_active = False
            
            # Sauvegarder le modèle final
            self._save_final_model()
            
            # Évaluation finale approfondie
            self._final_evaluation()
            
            # Statistiques finales
            training_time = time.time() - start_time
            self.custom_logger.info("🏁 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            self.custom_logger.info("=" * 80)
            self.custom_logger.info(f"⏱️  Temps total: {training_time/3600:.2f} heures")
            self.custom_logger.info(f"📊 Timesteps: {self.config['total_timesteps']:,}")
            self.custom_logger.info(f"📁 Résultats: {self.results_dir}")
            
        except KeyboardInterrupt:
            self.custom_logger.info("⏹️ Entraînement interrompu par l'utilisateur")
            self._save_final_model()
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur pendant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Nettoyage
            if self.env:
                try:
                    self.env.close()
                except:
                    pass            
                self.custom_logger.info("🧹 Nettoyage terminé")
    
    def _log_training_config(self):
        """Log la configuration complète"""
        self.custom_logger.info("⚙️ CONFIGURATION D'ENTRAÎNEMENT")
        self.custom_logger.info("-" * 50)
        
        config_items = [
            ("Algorithme", "TD3 (Twin Delayed DDPG)"),
            ("Total timesteps", f"{self.config['total_timesteps']:,}"),
            ("Learning rate", self.config.get('learning_rate', 3e-4)),
            ("Batch size", self.config.get('batch_size', 256)),
            ("Buffer size", f"{self.config.get('buffer_size', 1000000):,}"),
            ("Learning starts", f"{self.config.get('learning_starts', 25000):,}"),
            ("Tau (soft update)", self.config.get('tau', 0.005)),
            ("Gamma (discount)", self.config.get('gamma', 0.98)),
            ("Action noise sigma", self.config.get('action_noise_sigma', 0.1)),
            ("Architecture", self.config.get('net_arch', [400, 300])),
            ("Device", self.config.get('device', 'auto')),
            ("Max episode steps", self.config.get('max_episode_steps', 500)),
            ("Assistance activée", self.config.get('enable_assistance', True)),
            ("Évaluation freq.", f"{self.config.get('eval_freq', 25000):,}"),
            ("Vidéo freq.", f"{self.config.get('video_freq', 50000):,}"),
            ("Sauvegarde freq.", f"{self.config.get('save_freq', 50000):,}"),
        ]
        
        for key, value in config_items:
            self.custom_logger.info(f"  {key:<20}: {value}")
        
        self.custom_logger.info("-" * 50)
    
    def _save_final_model(self):
        """Sauvegarde le modèle final"""
        if not self.model:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Modèle final avec timestamp
            final_path = self.results_dir / f"final_model_{timestamp}.zip"
            self.model.save(str(final_path))
            
            # Modèle final (lien simple)
            latest_path = self.results_dir / "final_model.zip"
            self.model.save(str(latest_path))
            
            self.custom_logger.info(f"💾 Modèle final sauvegardé:")
            self.custom_logger.info(f"   Avec timestamp: {final_path}")
            self.custom_logger.info(f"   Dernière version: {latest_path}")
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur sauvegarde finale: {e}")
    
    def _final_evaluation(self):
        """Évaluation finale approfondie"""
        self.custom_logger.info("🔍 ÉVALUATION FINALE APPROFONDIE")
        self.custom_logger.info("-" * 50)
        
        try:
            # Créer un environnement d'évaluation propre
            eval_env = UltraRobustGraspEnv(
                render_mode="rgb_array",
                enable_assistance=False  # Évaluation sans assistance
            )
            
            n_episodes = 20
            results = {
                'episode_rewards': [],
                'episode_lengths': [],
                'final_distances': [],
                'best_distances': [],
                'contact_counts': [],
                'successful_grasps': 0
            }
            
            self.custom_logger.info(f"Évaluation sur {n_episodes} épisodes...")
            
            for episode in range(n_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                episode_length = 0
                best_distance = float('inf')
                max_contacts = 0
                
                for step in range(600):  # Episodes plus longs pour l'évaluation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Métriques
                    distance = info.get('distance', float('inf'))
                    contacts = info.get('contact_count', 0)
                    
                    if distance < best_distance:
                        best_distance = distance
                    
                    if contacts > max_contacts:
                        max_contacts = contacts
                    
                    if terminated or truncated:
                        break
                
                # Stocker les résultats
                results['episode_rewards'].append(episode_reward)
                results['episode_lengths'].append(episode_length)
                results['final_distances'].append(info.get('distance', float('inf')))
                results['best_distances'].append(best_distance)
                results['contact_counts'].append(max_contacts)
                
                # Grasp réussi si distance < 0.05 et au moins 2 contacts
                if best_distance < 0.05 and max_contacts >= 2:
                    results['successful_grasps'] += 1
                
                # Log périodique
                if (episode + 1) % 5 == 0:
                    self.custom_logger.info(f"   Épisodes {episode-4+1}-{episode+1} terminés")
            
            # Calculer les statistiques
            stats = {
                'mean_reward': np.mean(results['episode_rewards']),
                'std_reward': np.std(results['episode_rewards']),
                'mean_length': np.mean(results['episode_lengths']),
                'mean_final_distance': np.mean(results['final_distances']),
                'mean_best_distance': np.mean(results['best_distances']),
                'mean_contacts': np.mean(results['contact_counts']),
                'success_rate': results['successful_grasps'] / n_episodes * 100,
                'best_episode_reward': max(results['episode_rewards']),
                'best_distance_achieved': min(results['best_distances'])
            }
            
            # Afficher les résultats
            self.custom_logger.info("📊 RÉSULTATS FINAUX:")
            self.custom_logger.info(f"   Reward moyen: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            self.custom_logger.info(f"   Meilleur reward: {stats['best_episode_reward']:.2f}")
            self.custom_logger.info(f"   Longueur moyenne: {stats['mean_length']:.1f} steps")
            self.custom_logger.info(f"   Distance finale moyenne: {stats['mean_final_distance']:.3f} m")
            self.custom_logger.info(f"   Meilleure distance atteinte: {stats['best_distance_achieved']:.3f} m")
            self.custom_logger.info(f"   Contacts moyens max: {stats['mean_contacts']:.1f}")
            self.custom_logger.info(f"   Taux de succès: {stats['success_rate']:.1f}%")
            
            # Sauvegarder les résultats d'évaluation
            eval_results = {
                'statistics': stats,
                'raw_data': results,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            eval_file = self.results_dir / "final_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)
            
            self.custom_logger.info(f"💾 Résultats d'évaluation sauvegardés: {eval_file}")
            
            # Créer vidéo finale
            self._create_final_demo_video(eval_env)
            
            eval_env.close()
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur évaluation finale: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_final_demo_video(self, eval_env):
        """Crée la vidéo finale de démonstration"""
        if not imageio:
            self.custom_logger.warning("Imageio non disponible pour la vidéo finale")
            return
        
        self.custom_logger.info("🎬 Création de la vidéo finale de démonstration...")
        
        try:
            frames = []
            obs, _ = eval_env.reset()
            
            for step in range(500):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                # Capturer frame
                frame = eval_env.render()
                if frame is not None:
                    if isinstance(frame, np.ndarray):
                        frames.append(Image.fromarray(frame.astype(np.uint8)))
                    else:
                        frames.append(frame)
                
                if terminated or truncated:
                    break
            
            # Sauvegarder la vidéo
            if len(frames) > 10:
                video_path = self.results_dir / "final_demonstration.mp4"
                imageio.mimsave(str(video_path), frames, fps=30)
                self.custom_logger.info(f"🎥 Vidéo finale sauvegardée: {video_path}")
            else:
                self.custom_logger.warning("Pas assez de frames pour la vidéo finale")
            
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création vidéo finale: {e}")


def create_optimal_config():
    """Configuration optimale basée sur les insights du collègue"""
    return {
        # Entraînement principal
        'total_timesteps': 500_000,  # 500K pour des résultats robustes
        'learning_rate': 3e-4,       # Identique au collègue
        'batch_size': 256,           # Identique au collègue
        'buffer_size': 1_000_000,    # Identique au collègue
        'learning_starts': 25000,    # Plus conservateur
        'train_freq': 1,
        'gradient_steps': 1,
        
        # TD3 spécifique
        'tau': 0.02,                 # Identique au collègue
        'gamma': 0.98,               # Identique au collègue
        'action_noise_sigma': 0.1,   # Exploration modérée
        
        # Architecture
        'net_arch': [400, 300],       # Efficace et robuste
        
        # Environnement
        'max_episode_steps': 500,     # Identique au collègue
        'enable_assistance': True,    # Assistance contextuelle
        'render_mode': 'rgb_array',
        
        # Monitoring et sauvegarde
        'results_dir': 'ultra_robust_results',
        'log_freq': 1000,
        'eval_freq': 25000,          # Évaluation tous les 25K steps
        'video_freq': 50000,         # Vidéo tous les 50K steps
        'save_freq': 50000,          # Sauvegarde tous les 50K steps
        'n_eval_episodes': 5,
        'log_interval': 4,
        
        # Système
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Optionnel
        'load_model_path': None,      # Pour reprendre un entraînement
    }


def main():
    """Fonction principale"""
    print("🚀 ENTRAÎNEMENT ULTRA-ROBUSTE TD3 - INSIGHTS DU COLLÈGUE")
    print("=" * 80)
    print("🎯 IMPLÉMENTE TOUS LES INSIGHTS QUI FONCTIONNENT:")
    print("   ✅ Scaling adaptatif: ARM_SCALE = 0.4 si dist > 0.08 else 0.2")
    print("   ✅ Reset contrôles: self.data.ctrl[:] = 0.0 à chaque step")
    print("   ✅ Assistance contextuelle: aide quand 2+ doigts touchent")
    print("   ✅ Position cube fixe: [0.18, 0.0, 0.44]")
    print("   ✅ Rewards équilibrés: distance + contacts + qualité grasp")
    print("   ✅ Gestion robuste NaN/Inf")
    print("=" * 80)
    
    # Configuration optimale
    config = create_optimal_config()
    
    print(f"\n📊 CONFIGURATION ULTRA-ROBUSTE:")
    print(f"   Device: {config['device']}")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Architecture: {config['net_arch']}")
    print(f"   Buffer size: {config['buffer_size']:,}")
    print(f"   Assistance: {config['enable_assistance']}")
    print(f"   Évaluation chaque: {config['eval_freq']:,} steps")
    print("=" * 80)
    
    # Test préalable de l'environnement
    print("\n🧪 TEST PRÉALABLE DE L'ENVIRONNEMENT...")
    try:
        test_env = UltraRobustGraspEnv()
        obs, info = test_env.reset()
        
        for i in range(10):
            action = test_env.action_space.sample() * 0.3
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            # Vérifier NaN/Inf
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                raise ValueError(f"Observation NaN/Inf au step {i}")
            if np.isnan(reward) or np.isinf(reward):
                raise ValueError(f"Reward NaN/Inf au step {i}")
        
        test_env.close()
        print("✅ Test environnement réussi!")
        
    except Exception as e:
        print(f"❌ ERREUR dans le test environnement: {e}")
        print("L'entraînement ne peut pas continuer.")
        return False
    
    # Créer et lancer le trainer
    print("\n🚀 LANCEMENT DE L'ENTRAÎNEMENT...")
    trainer = UltraRobustTrainer(config)
    
    try:
        trainer.train()
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Tous les résultats sont dans: {trainer.results_dir}")
        print(f"🎥 Vidéo finale: {trainer.results_dir}/final_demonstration.mp4")
        print(f"💾 Modèle final: {trainer.results_dir}/final_model.zip")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)