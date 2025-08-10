#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT ROBUSTE TD3 - BASÉ SUR L'ANALYSE DU COLLÈGUE
==========================================================

Script d'entraînement robuste implémentant TOUS les insights du collègue:
✅ Scaling adaptatif selon distance
✅ Reset des contrôles à chaque step  
✅ Assistance contextuelle progressive
✅ Pure Reinforcement Learning (AUCUN CONTROL EXPLICITE)
✅ Phases smooth: approche → fixation palme → fermeture doigts

Objectif: ABOUTIR aux résultats comme le collègue mais de façon robuste
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# ML imports
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Notre environnement robuste
from envs.robust_smooth_grasp_env import RobustSmoothGraspEnv, GraspPhase

# Pour les vidéos
import imageio
from PIL import Image

class RobustSmoothCallback(BaseCallback):
 """
 Callback spécialisé pour l'entraînement robuste avec phases smooth
 """
 
 def __init__(self, eval_env, log_freq=1000, eval_freq=25000, 
              video_freq=50000, verbose=1):
     super().__init__(verbose)
     self.eval_env = eval_env
     self.log_freq = log_freq
     self.eval_freq = eval_freq
     self.video_freq = video_freq
     
     # Métriques de suivi
     self.phase_stats = {phase.value: {'episodes': 0, 'total_reward': 0} 
                        for phase in GraspPhase}
     self.smoothness_scores = []
     self.contact_success_rate = []
     
     # Dossier de résultats
     self.results_dir = Path("robust_smooth_results")
     self.results_dir.mkdir(exist_ok=True)
     (self.results_dir / "videos").mkdir(exist_ok=True)
 
 def _on_step(self) -> bool:
     # Log périodique des métriques
     if self.n_calls % self.log_freq == 0:
         try:
             env = self.training_env.get_env()
             info = env.get_wrapper_attr('_get_info')()
             
             # Log métriques de base
             self.logger.record("robust/current_phase", info['current_phase'])
             self.logger.record("robust/distance", info['distance'])
             self.logger.record("robust/contact_count", info['contact_count'])
             self.logger.record("robust/assistance_level", info['assistance_level'])
             self.logger.record("robust/smoothness_score", info['smoothness_score'])
             self.logger.record("robust/palm_stability", info['palm_stability'])
             
             # Suivi des phases
             current_phase = info['current_phase']
             if len(info['phase_history']) > 0:
                 self.logger.record("robust/phase_transitions", len(info['phase_history']))
             
             # Moyennes mobiles
             self.smoothness_scores.append(info['smoothness_score'])
             if len(self.smoothness_scores) > 100:
                 self.smoothness_scores.pop(0)
             
             avg_smoothness = np.mean(self.smoothness_scores) if self.smoothness_scores else 0
             self.logger.record("robust/avg_smoothness", avg_smoothness)
             
         except Exception as e:
             if self.verbose > 0:
                 print(f"⚠️ Erreur dans RobustSmoothCallback: {e}")
     
     # Évaluation périodique
     if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
         self._evaluate_model()
     
     # Vidéo périodique
     if self.n_calls % self.video_freq == 0 and self.n_calls > 0:
         self._create_evaluation_video()
     
     return True
 
 def _evaluate_model(self):
     """Évaluation robuste du modèle"""
     print(f"\n📊 Évaluation à {self.n_calls} steps...")
     
     episode_rewards = []
     phase_progressions = []
     smoothness_scores = []
     contact_successes = []
     
     for episode in range(5):
         obs, _ = self.eval_env.reset()
         episode_reward = 0
         episode_contacts = 0
         episode_smoothness = []
         
         for step in range(300):
             action, _ = self.model.predict(obs, deterministic=True)
             obs, reward, terminated, truncated, info = self.eval_env.step(action)
             episode_reward += reward
             
             # Métriques
             if info['contact_count'] > 0:
                 episode_contacts += 1
             episode_smoothness.append(info['smoothness_score'])
             
             if terminated or truncated:
                 break
         
         episode_rewards.append(episode_reward)
         phase_progressions.append(len(info['phase_history']))
         smoothness_scores.append(np.mean(episode_smoothness))
         contact_successes.append(episode_contacts > 10)  # Au moins 10 steps avec contact
     
     # Log des résultats
     avg_reward = np.mean(episode_rewards)
     avg_phases = np.mean(phase_progressions)
     avg_smoothness = np.mean(smoothness_scores)
     contact_rate = np.mean(contact_successes) * 100
     
     print(f"   Reward moyen: {avg_reward:.2f}")
     print(f"   Transitions moyennes: {avg_phases:.1f}")
     print(f"   Smoothness moyen: {avg_smoothness:.2f}")
     print(f"   Taux de contact: {contact_rate:.1f}%")
     
     # Log TensorBoard
     self.logger.record("eval/mean_reward", avg_reward)
     self.logger.record("eval/mean_phases", avg_phases)
     self.logger.record("eval/mean_smoothness", avg_smoothness)
     self.logger.record("eval/contact_success_rate", contact_rate)
     
     # Sauvegarder le modèle si bon performance
     if avg_reward > getattr(self, 'best_eval_reward', -float('inf')):
         self.best_eval_reward = avg_reward
         model_path = self.results_dir / f"best_model_{self.n_calls}steps.zip"
         self.model.save(model_path)
         print(f"💾 Nouveau meilleur modèle sauvegardé: {model_path}")
 
 def _create_evaluation_video(self):
     """Créer vidéo d'évaluation montrant les phases"""
     print(f"🎥 Création vidéo d'évaluation...")
     
     obs, _ = self.eval_env.reset()
     frames = []
     
     for step in range(400):
         action, _ = self.model.predict(obs, deterministic=True)
         obs, reward, terminated, truncated, info = self.eval_env.step(action)
         
         # Capturer frame si possible
         frame = self.eval_env.render()
         if frame is not None:
             frames.append(Image.fromarray(frame.astype(np.uint8)))
         
         if terminated or truncated:
             obs, _ = self.eval_env.reset()
     
     # Sauvegarder vidéo
     if frames:
         timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
         video_path = self.results_dir / "videos" / f"eval_{self.n_calls}steps_{timestamp}.mp4"
         imageio.mimsave(video_path, frames, fps=30)
         print(f"✅ Vidéo sauvegardée: {video_path}")

class RobustTrainer:
 """
 Entraîneur robuste basé sur l'analyse du collègue
 """
 
 def __init__(self, config):
     self.config = config
     self.results_dir = Path(config['results_dir'])
     self.results_dir.mkdir(exist_ok=True)
     
     print("🤖 ENTRAÎNEUR ROBUSTE BASÉ SUR L'ANALYSE DU COLLÈGUE")
     print(f"📁 Répertoire: {self.results_dir}")
 
 def create_environments(self):
     """Créer environnements robustes"""
     print("🏗️ Création des environnements robustes...")
     
     # Environnement d'entraînement avec assistance progressive
     train_env = RobustSmoothGraspEnv(
         auto_phase_progression=True,
         initial_assistance_level=self.config.get('initial_assistance', 0.6)
     )
     
     # Wrap avec Monitor pour le logging
     self.train_env = Monitor(train_env, filename=str(self.results_dir / "train_monitor.csv"))
     
     # Environnement d'évaluation sans assistance progressive
     self.eval_env = RobustSmoothGraspEnv(
         auto_phase_progression=True,
         initial_assistance_level=0.0  # Pas d'assistance pour l'éval
     )
     
     print(f"✅ Environnements créés")
     print(f"   Action space: {self.train_env.action_space.shape}")
     print(f"   Observation space: {self.train_env.observation_space.shape}")
     print(f"   Assistance initiale train: {self.config.get('initial_assistance', 0.6)}")
     print(f"   Assistance éval: 0.0 (autonome)")
     
     return self.train_env, self.eval_env
 
 def create_model(self, env):
     """Créer modèle TD3 robuste avec config du collègue"""
     print("🧠 Création du modèle TD3 robuste...")
     
     # Action noise adapté pour les mouvements smooth
     n_actions = env.action_space.shape[-1]
     action_noise = NormalActionNoise(
         mean=np.zeros(n_actions),
         sigma=self.config.get('action_noise_sigma', 0.08) * np.ones(n_actions)  # Plus doux
     )
     
     # Modèle TD3 avec hyperparamètres optimisés
     model = TD3(
         "MlpPolicy",
         env,
         learning_rate=self.config.get('learning_rate', 3e-4),
         buffer_size=self.config.get('buffer_size', 500_000),  # Plus petit pour plus d'efficacité
         learning_starts=self.config.get('learning_starts', 10000),
         batch_size=self.config.get('batch_size', 256),
         tau=self.config.get('tau', 0.02),  # Comme le collègue
         gamma=self.config.get('gamma', 0.98),
         train_freq=self.config.get('train_freq', 1),
         gradient_steps=self.config.get('gradient_steps', 1),
         action_noise=action_noise,
         policy_kwargs={
             'net_arch': self.config.get('net_arch', [400, 300])  # Plus compact
         },
         verbose=1,
         device=self.config.get('device', 'auto'),
         tensorboard_log=str(self.results_dir / "tensorboard")
     )
     
     print(f"✅ Modèle TD3 créé")
     print(f"   Device: {model.device}")
     print(f"   Learning rate: {model.learning_rate}")
     print(f"   Buffer size: {model.buffer_size:,}")
     print(f"   Action noise sigma: {self.config.get('action_noise_sigma', 0.08)}")
     
     return model
 
 def train(self):
     """Lancement de l'entraînement robuste"""
     print("\n🚀 DÉMARRAGE ENTRAÎNEMENT ROBUSTE")
     print("=" * 60)
     print("✅ Implémente TOUS les insights du collègue qui fonctionne")
     print("✅ Phases smooth: approche → palme → doigts")  
     print("✅ Pure RL avec assistance progressive qui diminue")
     print("=" * 60)
     
     # Créer environnements et modèle
     train_env, eval_env = self.create_environments()
     model = self.create_model(train_env)
     
     # Callback robuste
     callback = RobustSmoothCallback(
         eval_env=eval_env,
         log_freq=self.config.get('log_freq', 1000),
         eval_freq=self.config.get('eval_freq', 25000),
         video_freq=self.config.get('video_freq', 50000)
     )
     
     # Configuration
     total_timesteps = self.config.get('total_timesteps', 300_000)
     
     print(f"\n📊 Configuration d'entraînement:")
     print(f"   Total timesteps: {total_timesteps:,}")
     print(f"   Évaluation chaque: {self.config.get('eval_freq', 25000):,} steps")
     print(f"   Vidéo chaque: {self.config.get('video_freq', 50000):,} steps")
     print(f"   Phases automatiques: activées")
     print(f"   Scaling adaptatif: activé (comme le collègue)")
     print("\n🏃 Entraînement en cours...\n")
     
     start_time = time.time()
     
     try:
         # Entraînement
         model.learn(
             total_timesteps=total_timesteps,
             callback=callback,
             log_interval=self.config.get('log_interval', 4),
             progress_bar=True
         )
         
         training_time = time.time() - start_time
         
         # Sauvegarder modèle final
         final_model_path = self.results_dir / "final_robust_model.zip"
         model.save(final_model_path)
         
         print("\n" + "="*60)
         print("🎉 ENTRAÎNEMENT ROBUSTE TERMINÉ")
         print("="*60)
         print(f"⏱️  Durée: {training_time/60:.1f} minutes")
         print(f"💾 Modèle final: {final_model_path}")
         print(f"📁 Résultats: {self.results_dir}")
         
         # Évaluation finale complète
         self._final_comprehensive_evaluation(model, eval_env)
         
     except KeyboardInterrupt:
         print("\n⏹️ Entraînement interrompu")
         model.save(self.results_dir / "interrupted_robust_model.zip")
         
     except Exception as e:
         print(f"\n❌ Erreur: {e}")
         
     finally:
         # Fermeture propre
         train_env.close()
         eval_env.close()
         print("✅ Environnements fermés")
 
 def _final_comprehensive_evaluation(self, model, eval_env):
     """Évaluation finale complète par phase"""
     print("\n📊 ÉVALUATION FINALE COMPLÈTE")
     print("-" * 40)
     
     total_episodes = 10
     results = {
         'episodes': total_episodes,
         'rewards': [],
         'phase_progressions': [],
         'smoothness_scores': [],
         'contact_rates': [],
         'success_rates': []
     }
     
     for episode in range(total_episodes):
         obs, _ = eval_env.reset()
         episode_reward = 0
         contact_steps = 0
         smoothness_scores = []
         successful_contacts = False
         
         for step in range(400):
             action, _ = model.predict(obs, deterministic=True)
             obs, reward, terminated, truncated, info = eval_env.step(action)
             episode_reward += reward
             
             # Métriques
             if info['contact_count'] > 0:
                 contact_steps += 1
             if info['contact_count'] >= 2:
                 successful_contacts = True
             
             smoothness_scores.append(info['smoothness_score'])
             
             if terminated or truncated:
                 break
         
         # Stocker résultats
         results['rewards'].append(episode_reward)
         results['phase_progressions'].append(len(info['phase_history']))
         results['smoothness_scores'].append(np.mean(smoothness_scores))
         results['contact_rates'].append(contact_steps / (step + 1) * 100)
         results['success_rates'].append(successful_contacts)
     
     # Afficher résultats
     print(f"📈 Résultats sur {total_episodes} épisodes:")
     print(f"   Reward moyen: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
     print(f"   Progressions phase: {np.mean(results['phase_progressions']):.1f} ± {np.std(results['phase_progressions']):.1f}")
     print(f"   Smoothness moyen: {np.mean(results['smoothness_scores']):.3f} ± {np.std(results['smoothness_scores']):.3f}")
     print(f"   Taux contact moyen: {np.mean(results['contact_rates']):.1f}%")
     print(f"   Taux succès contacts: {np.mean(results['success_rates']) * 100:.1f}%")
     
     # Vidéo finale de démonstration
     self._create_final_demo_video(model, eval_env)
 
 def _create_final_demo_video(self, model, eval_env):
     """Créer vidéo finale de démonstration"""
     print("\n🎬 Création vidéo finale...")
     
     obs, _ = eval_env.reset()
     frames = []
     phase_changes = []
     
     for step in range(500):
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, terminated, truncated, info = eval_env.step(action)
         
         # Capturer les changements de phase
         if len(info['phase_history']) > len(phase_changes):
             phase_changes.extend(info['phase_history'][len(phase_changes):])
         
         # Capturer frame
         frame = eval_env.render()
         if frame is not None:
             frames.append(Image.fromarray(frame.astype(np.uint8)))
         
         if terminated or truncated:
             break
     
     # Sauvegarder
     if frames:
         video_path = self.results_dir / "final_robust_demo.mp4"
         imageio.mimsave(video_path, frames, fps=30)
         print(f"✅ Vidéo finale: {video_path}")
         print(f"   Transitions de phase: {len(phase_changes)}")

def create_robust_config():
 """Configuration robuste basée sur l'analyse du collègue"""
 return {
     # Entraînement principal
     'total_timesteps': 300_000,  # Plus court mais efficace
     'learning_rate': 3e-4,      # Comme le collègue
     'batch_size': 256,
     'buffer_size': 500_000,     # Plus compact
     'learning_starts': 10_000,   # Plus rapide
     
     # TD3 spécifique
     'tau': 0.02,                # Comme le collègue 
     'gamma': 0.98,
     'train_freq': 1,
     'gradient_steps': 1,
     'action_noise_sigma': 0.08,  # Plus doux pour mouvements smooth
     
     # Architecture
     'net_arch': [400, 300],      # Plus compact mais efficace
     
     # Assistance progressive
     'initial_assistance': 0.6,   # Aide initiale modérée
     
     # Monitoring
     'results_dir': 'robust_smooth_results',
     'log_freq': 1000,
     'eval_freq': 25000,
     'video_freq': 50000,
     'log_interval': 4,
     
     # Système
     'device': 'auto',
 }

def main():
 """Fonction principale"""
 print("🤖 ENTRAÎNEMENT ROBUSTE TD3 - INSIGHTS DU COLLÈGUE")
 print("=" * 60)
 print("🎯 Objectif: ABOUTIR aux résultats avec les phases smooth")
 print("🎯 Stratégie: Scaling adaptatif + assistance progressive")
 print("🎯 Phases: Approche smooth → Fixation palme → Fermeture doigts")
 print("=" * 60)
 
 # Configuration basée sur l'analyse
 config = create_robust_config()
 
 print(f"\n📊 Configuration robuste:")
 print(f"   Total timesteps: {config['total_timesteps']:,}")
 print(f"   Assistance initiale: {config['initial_assistance']}")
 print(f"   Action noise: {config['action_noise_sigma']} (smooth)")
 print(f"   Architecture: {config['net_arch']}")
 print(f"   Scaling adaptatif: ✅ (insight clé du collègue)")
 print(f"   Reset contrôles: ✅ (insight clé du collègue)")
 print(f"   Assistance contextuelle: ✅ (insight clé du collègue)")
 print()
 
 # Créer et lancer l'entraîneur
 trainer = RobustTrainer(config)
 trainer.train()

if __name__ == "__main__":
 main()
