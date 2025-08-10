#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT PROFESSIONNEL TD3 AVEC CURRICULUM ADAPTATIF
==========================================================

Système d'entraînement professionnel avec:
- Curriculum learning adaptatif par étapes
- Système de récompenses sophistiqué
- Progression automatique
- Monitoring avancé
- Aboutir aux RÉSULTATS, pas juste "fonctionner"
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ML imports
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Notre environnement professionnel
from envs.professional_grasp_env import ProfessionalGraspEnv, TrainingStage

# Pour les vidéos
import imageio
from PIL import Image

class CurriculumProgressionCallback(BaseCallback):
    """
    Callback spécialisé pour gérer la progression du curriculum
    """
    
    def __init__(self, env_wrapper, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.log_freq = log_freq
        self.stage_transitions = []
        self.performance_history = []
        
    def _on_step(self) -> bool:
        # Vérifier si l'environnement a changé d'étape
        current_env = self.env_wrapper.get_env()
        current_stage = current_env.current_stage
        
        # Log périodique des métriques
        if self.n_calls % self.log_freq == 0:
            try:
                stats = current_env.get_training_statistics()
                
                self.logger.record("curriculum/current_stage", current_stage.value)
                self.logger.record("curriculum/stage_episodes", stats['stage_episodes'])
                self.logger.record("curriculum/total_episodes", stats['total_episodes'])
                
                if stats['stage_rewards']:
                    avg_recent = np.mean(stats['stage_rewards'][-10:])
                    self.logger.record("curriculum/avg_recent_reward", avg_recent)
                
                self.logger.record("curriculum/avg_contact_count", stats['avg_contact_count'])
                self.logger.record("curriculum/avg_grasp_quality", stats['avg_grasp_quality'])
                
                # Détection de transitions
                if len(stats['curriculum_transitions']) > len(self.stage_transitions):
                    new_transitions = stats['curriculum_transitions'][len(self.stage_transitions):]
                    self.stage_transitions.extend(new_transitions)
                    
                    for transition in new_transitions:
                        print(f"🎓 CURRICULUM PROGRESSION: {transition['from_stage']} → {transition['to_stage']} "
                                f"(épisode {transition['episode']})")
                        
                        self.logger.record("curriculum/transition_episode", transition['episode'])
                        self.logger.record("curriculum/transition_timestamp", transition['timestamp'])
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Erreur dans CurriculumProgressionCallback: {e}")
        
        return True

class ProfessionalVideoCallback(BaseCallback):
    """
    Callback pour enregistrer des vidéos d'évaluation sophistiquées
    """
    
    def __init__(self, eval_env, eval_freq=50000, video_length=500, 
                video_folder="professional_videos/", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.video_length = video_length
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            current_stage = self.eval_env.current_stage
            stage_name = current_stage.value
            
            print(f"🎥 Enregistrement vidéo d'évaluation - Étape: {stage_name}")
            
            obs, _ = self.eval_env.reset()
            frames = []
            
            episode_reward = 0
            contacts_detected = 0
            successful_grasps = 0
            
            for step in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                
                # Statistiques
                if info.get('contact_count', 0) > 0:
                    contacts_detected += 1
                if info.get('successful_grasp', False):
                    successful_grasps += 1
                
                # Capturer frame
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame.astype(np.uint8)))
                
                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
            
            # Sauvegarder la vidéo
            if frames:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                video_name = f"eval_{stage_name}_{self.n_calls}steps_{timestamp}.mp4"
                video_path = self.video_folder / video_name
                
                imageio.mimsave(video_path, frames, fps=30)
                
                print(f"✅ Vidéo sauvegardée: {video_path}")
                print(f"   Reward total: {episode_reward:.1f}")
                print(f"   Contacts détectés: {contacts_detected}/{self.video_length}")
                print(f"   Grasps réussis: {successful_grasps}")
                
                # Log des métriques
                self.logger.record("video/episode_reward", episode_reward)
                self.logger.record("video/contact_ratio", contacts_detected / self.video_length)
                self.logger.record("video/success_count", successful_grasps)
        
        return True

class StageSpecificMonitor(BaseCallback):
    """
    Monitoring spécialisé par étape d'entraînement
    """
    
    def __init__(self, results_dir="professional_results", save_freq=25000):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.save_freq = save_freq
        
        # Métriques par étape
        self.stage_metrics = {}
        self.current_stage = None
        
    def _on_step(self) -> bool:
        try:
            # Obtenir l'environnement et ses métriques
            env = self.training_env.get_env()
            current_stage = env.current_stage.value
            
            # Initialiser les métriques pour une nouvelle étape
            if current_stage != self.current_stage:
                if current_stage not in self.stage_metrics:
                    self.stage_metrics[current_stage] = {
                        'rewards': [],
                        'contacts': [],
                        'grasp_qualities': [],
                        'episodes': 0,
                        'start_step': self.n_calls
                    }
                self.current_stage = current_stage
            
            # Sauvegarder périodiquement
            if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
                self._save_comprehensive_stats(env)
                
                # Sauvegarder le modèle avec nom spécifique à l'étape
                model_name = f"model_{current_stage}_{self.n_calls}steps.zip"
                model_path = self.results_dir / model_name
                self.model.save(model_path)
                print(f"💾 Modèle sauvegardé (étape {current_stage}): {model_path}")
        
        except Exception as e:
            print(f"⚠️ Erreur dans StageSpecificMonitor: {e}")
        
        return True
    
    def _save_comprehensive_stats(self, env):
        """Sauvegarder des statistiques complètes"""
        try:
            env_stats = env.get_training_statistics()
            
            comprehensive_stats = {
                'timestamp': datetime.now().isoformat(),
                'total_steps': self.n_calls,
                'environment_stats': env_stats,
                'stage_metrics': self.stage_metrics,
                'curriculum_progression': {
                    'current_stage': env.current_stage.value,
                    'total_transitions': len(env_stats.get('curriculum_transitions', [])),
                    'stages_completed': len([t for t in env_stats.get('curriculum_transitions', [])]),
                }
            }
            
            stats_file = self.results_dir / f"comprehensive_stats_{self.n_calls}.json"
            with open(stats_file, 'w') as f:
                json.dump(comprehensive_stats, f, indent=2, default=str)
            
            print(f"📊 Statistiques complètes sauvegardées: {stats_file}")
            
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde statistiques: {e}")

class ProfessionalTrainer:
    """
    Entraîneur professionnel avec curriculum adaptatif
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration de logging
        self._setup_logging()
        
        print("🎯 ENTRAÎNEUR PROFESSIONNEL INITIALISÉ")
        print(f"📁 Répertoire: {self.results_dir}")
    
    def _setup_logging(self):
        """Configuration du logging professionnel"""
        import logging
        
        self.logger = logging.getLogger("ProfessionalTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Handler pour fichier
        log_file = self.results_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler pour console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def create_environments(self):
        """Créer les environnements d'entraînement et d'évaluation"""
        print("🏗️ Création des environnements professionnels...")
        
        # Environnement d'entraînement
        train_env = ProfessionalGraspEnv(
            stage=TrainingStage.STAGE_1_APPROACH,
            auto_progression=True
        )
        self.train_env = Monitor(train_env)
        
        # Environnement d'évaluation
        self.eval_env = ProfessionalGraspEnv(
            stage=TrainingStage.STAGE_1_APPROACH,
            auto_progression=False  # Pas de progression automatique pour l'éval
        )
        
        print(f"✅ Environnements créés")
        print(f"   Action space: {self.train_env.action_space.shape}")
        print(f"   Observation space: {self.train_env.observation_space.shape}")
        
        return self.train_env, self.eval_env
    
    def create_model(self, env):
        """Créer le modèle TD3 professionnel"""
        print("🧠 Création du modèle TD3 professionnel...")
        
        # Action noise adaptatif
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.config.get('action_noise_sigma', 0.1) * np.ones(n_actions)
        )
        
        # Créer le modèle
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            buffer_size=self.config.get('buffer_size', 1_000_000),
            learning_starts=self.config.get('learning_starts', 25000),  # Plus long pour curriculum
            batch_size=self.config.get('batch_size', 256),
            tau=self.config.get('tau', 0.02),
            gamma=self.config.get('gamma', 0.99),
            train_freq=self.config.get('train_freq', 1),
            gradient_steps=self.config.get('gradient_steps', 1),
            action_noise=action_noise,
            policy_kwargs={
                'net_arch': self.config.get('net_arch', [512, 512, 256])
            },
            verbose=1,
            device=self.config.get('device', 'auto'),
            tensorboard_log=str(self.results_dir / "tensorboard")
        )
        
        print(f"✅ Modèle TD3 créé")
        print(f"   Device: {model.device}")
        print(f"   Learning rate: {model.learning_rate}")
        print(f"   Buffer size: {model.buffer_size:,}")
        print(f"   Architecture: {self.config.get('net_arch', [512, 512, 256])}")
        
        return model
    
    def setup_callbacks(self, env, eval_env):
        """Configurer les callbacks professionnels"""
        callbacks = []
        
        # Curriculum progression
        curriculum_cb = CurriculumProgressionCallback(
            env_wrapper=env,
            log_freq=self.config.get('log_freq', 1000)
        )
        callbacks.append(curriculum_cb)
        
        # Monitoring par étape
        monitor_cb = StageSpecificMonitor(
            results_dir=self.results_dir,
            save_freq=self.config.get('save_freq', 25000)
        )
        callbacks.append(monitor_cb)
        
        # Vidéos d'évaluation
        if self.config.get('record_videos', True):
            video_cb = ProfessionalVideoCallback(
                eval_env=eval_env,
                eval_freq=self.config.get('video_freq', 50000),
                video_length=self.config.get('video_length', 500),
                video_folder=str(self.results_dir / "videos")
            )
            callbacks.append(video_cb)
        
        return callbacks
    
    def train(self):
        """Lancement de l'entraînement professionnel"""
        print("\n🚀 DÉMARRAGE ENTRAÎNEMENT PROFESSIONNEL")
        print("=" * 80)
        
        # Créer les environnements
        train_env, eval_env = self.create_environments()
        
        # Créer le modèle
        model = self.create_model(train_env)
        
        # Configurer les callbacks
        callbacks = self.setup_callbacks(train_env, eval_env)
        
        # Configuration de l'entraînement
        total_timesteps = self.config.get('total_timesteps', 500_000)
        
        print(f"📊 Configuration d'entraînement:")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Curriculum: Progression automatique activée")
        print(f"   Étape initiale: {train_env.get_env().current_stage.value}")
        print(f"   Callbacks: {len(callbacks)} callbacks actifs")
        print("\n🏃 Entraînement en cours...\n")
        
        # Sauvegarder la configuration
        config_file = self.results_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        start_time = time.time()
        
        try:
            # Entraînement
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=self.config.get('log_interval', 10),
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            
            # Sauvegarder le modèle final
            final_model_path = self.results_dir / "final_professional_model.zip"
            model.save(final_model_path)
            
            # Statistiques finales
            final_stats = train_env.get_env().get_training_statistics()
            
            print("\n" + "="*80)
            print("🎉 ENTRAÎNEMENT PROFESSIONNEL TERMINÉ")
            print("="*80)
            print(f"⏱️  Durée totale: {training_time/3600:.1f} heures")
            print(f"📊 Étape finale: {final_stats['current_stage']}")
            print(f"🎓 Transitions: {len(final_stats.get('curriculum_transitions', []))}")
            print(f"📈 Épisodes totaux: {final_stats['total_episodes']}")
            print(f"💾 Modèle final: {final_model_path}")
            print(f"📁 Résultats: {self.results_dir}")
            
            # Évaluation finale
            self._final_evaluation(model, eval_env)
            
        except KeyboardInterrupt:
            print("\n⏹️ Entraînement interrompu par l'utilisateur")
            model.save(self.results_dir / "interrupted_professional_model.zip")
            
        except Exception as e:
            print(f"\n❌ Erreur pendant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Fermeture propre
            train_env.close()
            eval_env.close()
            print("✅ Environnements fermés proprement")
    
    def _final_evaluation(self, model, eval_env):
        """Évaluation finale complète"""
        print("\n📊 ÉVALUATION FINALE COMPLÈTE")
        print("-" * 50)
        
        # Tester sur chaque étape
        for stage in TrainingStage:
            print(f"\n🎯 Test étape: {stage.value}")
            
            eval_env.set_stage(stage)
            
            episode_rewards = []
            success_count = 0
            
            for episode in range(5):
                obs, _ = eval_env.reset()
                episode_reward = 0
                
                for step in range(200):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                if info.get('successful_grasp', False):
                    success_count += 1
            
            avg_reward = np.mean(episode_rewards)
            success_rate = success_count / 5 * 100
            
            print(f"   Reward moyen: {avg_reward:.2f}")
            print(f"   Taux de succès: {success_rate:.1f}%")
        
        # Créer une vidéo finale
        self._create_final_showcase_video(model, eval_env)
    
    def _create_final_showcase_video(self, model, eval_env):
        """Créer une vidéo finale de démonstration"""
        print("\n🎬 Création de la vidéo finale de démonstration...")
        
        frames = []
        
        # Démonstration sur chaque étape
        for stage in TrainingStage:
            eval_env.set_stage(stage)
            obs, _ = eval_env.reset()
            
            # 100 steps par étape
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                frame = eval_env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame.astype(np.uint8)))
                
                if terminated or truncated:
                    obs, _ = eval_env.reset()
        
        # Sauvegarder
        if frames:
            video_path = self.results_dir / "final_professional_showcase.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"✅ Vidéo finale créée: {video_path}")

def create_professional_config():
    """Configuration professionnelle optimisée"""
    return {
        # Entraînement principal
        'total_timesteps': 1_000_000,  # 1M pour curriculum complet
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 1_000_000,
        'learning_starts': 25000,  # Plus long pour le curriculum
        
        # Hyperparamètres TD3
        'tau': 0.02,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'action_noise_sigma': 0.1,
        
        # Architecture
        'net_arch': [512, 512, 256],
        
        # Monitoring et sauvegarde
        'results_dir': 'professional_td3_results',
        'save_freq': 50000,
        'log_freq': 1000,
        'log_interval': 10,
        
        # Vidéos
        'record_videos': True,
        'video_freq': 100000,
        'video_length': 500,
        
        # Système
        'device': 'auto',
  }
def main():
    """Fonction principale"""
    print("🎯 ENTRAÎNEMENT PROFESSIONNEL TD3 AVEC CURRICULUM")
    print("=" * 80)
    print("Objectif: ABOUTIR aux résultats, pas juste fonctionner")
    print("Stratégie: Progression par étapes avec rewards sophistiqués")
    print("=" * 80)
    
    # Configuration
    config = create_professional_config()
    
    print(f"📊 Configuration professionnelle:")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Buffer size: {config['buffer_size']:,}")
    print(f"   Architecture: {config['net_arch']}")
    print(f"   Device: {config['device']}")
    print()
    
    # Créer et lancer l'entraîneur
    trainer = ProfessionalTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
