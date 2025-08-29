#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT TD3 SIMPLIFIÉ ET ROBUSTE
=======================================

Script d'entraînement basé sur le code fonctionnel du collègue,
simplifié pour éviter la stagnation des rewards et les erreurs.

Inspiré du travail de votre collègue qui utilise TD3 avec succès.
"""

import os
import sys
import numpy as np
import torch
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Imports ML
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import de notre environnement simplifié
from envs.simple_robust_grasp_env import SimpleRobustGraspEnv

# Imports pour les vidéos (comme le collègue)
import imageio
from PIL import Image

class EvalVideoCallback(BaseCallback):
 """
 Callback pour enregistrer des vidéos d'évaluation
 Basé sur le code du collègue
 """
 def __init__(self, eval_env, eval_freq=25000, video_length=300, 
              video_folder="videos/", prefix="grasp_eval", verbose=1):
     super().__init__(verbose)
     self.eval_env = eval_env
     self.eval_freq = eval_freq
     self.video_length = video_length
     self.video_folder = video_folder
     self.prefix = prefix
     os.makedirs(video_folder, exist_ok=True)

 def _on_step(self) -> bool:
     if self.n_calls % self.eval_freq == 0:
         print(f"🎥 Enregistrement vidéo d'évaluation (step {self.n_calls})...")
         
         obs, _ = self.eval_env.reset()
         frames = []

         for _ in range(self.video_length):
             action, _ = self.model.predict(obs, deterministic=True)
             obs, _, done, _, _ = self.eval_env.step(action)

             # Enregistrer la frame
             frame = self.eval_env.render()
             if frame is not None:
                 frames.append(Image.fromarray(frame.astype(np.uint8)))

             if done:
                 break

         # Sauvegarder la vidéo
         if frames:
             timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
             video_path = os.path.join(
                 self.video_folder, f"{self.prefix}_{self.n_calls}_steps_{timestamp}.mp4"
             )
             imageio.mimsave(video_path, frames, fps=30)
             print(f"🎥 Vidéo sauvegardée: {video_path}")

     return True

class TrainingMonitor(BaseCallback):
 """Callback de monitoring pendant l'entraînement"""
 
 def __init__(self, log_freq=1000, save_freq=25000, results_dir="results"):
     super().__init__()
     self.log_freq = log_freq
     self.save_freq = save_freq
     self.results_dir = Path(results_dir)
     self.results_dir.mkdir(exist_ok=True)
     
     # Statistiques
     self.episode_rewards = []
     self.episode_lengths = []
     self.current_episode_reward = 0
     self.current_episode_length = 0
     
 def _on_step(self) -> bool:
     # Accumuler les rewards
     try:
         reward = self.locals.get('rewards', [0])[0]
         self.current_episode_reward += reward
         self.current_episode_length += 1
         
         # Détecter fin d'épisode
         done = self.locals.get('dones', [False])[0]
         if done:
             self.episode_rewards.append(self.current_episode_reward)
             self.episode_lengths.append(self.current_episode_length)
             
             # Log périodique
             if len(self.episode_rewards) % 10 == 0:
                 mean_reward = np.mean(self.episode_rewards[-10:])
                 print(f"Episode {len(self.episode_rewards)}: Reward moyen (10 derniers) = {mean_reward:.2f}")
             
             # Reset pour prochain épisode
             self.current_episode_reward = 0
             self.current_episode_length = 0
         
         # Sauvegarde périodique du modèle
         if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
             model_path = self.results_dir / f"model_{self.n_calls}_steps.zip"
             self.model.save(model_path)
             print(f"💾 Modèle sauvegardé: {model_path}")
             
             # Sauvegarder les statistiques
             stats = {
                 'episode_rewards': self.episode_rewards,
                 'episode_lengths': self.episode_lengths,
                 'total_steps': self.n_calls,
                 'timestamp': datetime.now().isoformat()
             }
             stats_path = self.results_dir / f"stats_{self.n_calls}_steps.json"
             with open(stats_path, 'w') as f:
                 json.dump(stats, f, indent=2)
                 
     except Exception as e:
         print(f"⚠️ Erreur dans TrainingMonitor: {e}")
     
     return True

def create_environment(eval_mode=False):
 """Créer un environnement avec monitoring"""
 env = SimpleRobustGraspEnv(eval_mode=eval_mode)
 if not eval_mode:
     env = Monitor(env)
 return env

def train_td3_robot():
 """
 Entraîner le robot avec TD3
 Configuration basée sur le code du collègue
 """
 print("🚀 DÉMARRAGE ENTRAÎNEMENT TD3 SIMPLIFIÉ")
 print("=" * 50)
 
 # Configuration
 total_timesteps = 100_000  # Commencer plus petit pour tester
 learning_rate = 3e-4
 batch_size = 256
 buffer_size = 1_000_000
 
 # Créer les répertoires
 results_dir = Path("simple_td3_results")
 video_dir = results_dir / "videos"
 results_dir.mkdir(exist_ok=True)
 video_dir.mkdir(exist_ok=True)
 
 # Environnements
 print("🏗️ Création des environnements...")
 env = create_environment(eval_mode=False)
 eval_env = create_environment(eval_mode=True)
 
 print(f"✅ Environnement créé: Action space {env.action_space.shape}")
 print(f"✅ Observation space: {env.observation_space.shape}")
 
 # Configurer le logger
 logger = configure(str(results_dir / "logs"), ["stdout", "csv", "tensorboard"])
 
 # Action noise (comme dans le code du collègue)
 n_actions = env.action_space.shape[-1]
 action_noise = NormalActionNoise(
     mean=np.zeros(n_actions), 
     sigma=0.1 * np.ones(n_actions)
 )
 
 # Créer le modèle TD3
 print("🧠 Création du modèle TD3...")
 model = TD3(
     "MlpPolicy",
     env,
     learning_rate=learning_rate,
     buffer_size=buffer_size,
     learning_starts=10000,
     batch_size=batch_size,
     tau=0.02,  # Comme le collègue
     gamma=0.98,  # Comme le collègue
     train_freq=1,
     gradient_steps=1,
     action_noise=action_noise,
     verbose=1,
     device='cuda' if torch.cuda.is_available() else 'cpu'
 )
 
 # Configurer le logger
 model.set_logger(logger)
 
 print(f"✅ Modèle TD3 créé sur device: {model.device}")
 
 # Callbacks
 monitor_callback = TrainingMonitor(
     log_freq=1000,
     save_freq=25000,
     results_dir=results_dir
 )
 
 video_callback = EvalVideoCallback(
     eval_env=eval_env,
     eval_freq=25000,
     video_length=300,
     video_folder=str(video_dir),
     prefix="td3_grasp_eval"
 )
 
 # Lancer l'entraînement
 print("🏃 Démarrage de l'entraînement...")
 print(f"📊 Total timesteps: {total_timesteps:,}")
 print(f"💾 Buffer size: {buffer_size:,}")
 print(f"🎯 Batch size: {batch_size}")
 
 start_time = time.time()
 
 try:
     model.learn(
         total_timesteps=total_timesteps,
         callback=[monitor_callback, video_callback],
         log_interval=10,
         progress_bar=True
     )
     
     training_time = time.time() - start_time
     print(f"✅ Entraînement terminé en {training_time:.1f} secondes")
     
     # Sauvegarder le modèle final
     final_model_path = results_dir / "final_model.zip"
     model.save(final_model_path)
     print(f"💾 Modèle final sauvegardé: {final_model_path}")
     
     return model, env, eval_env
     
 except KeyboardInterrupt:
     print("\n⏹️ Entraînement interrompu par l'utilisateur")
     model.save(results_dir / "interrupted_model.zip")
     return model, env, eval_env
 
 except Exception as e:
     print(f"❌ Erreur pendant l'entraînement: {e}")
     import traceback
     traceback.print_exc()
     return None, env, eval_env
     
def evaluate_and_create_video(model, env, video_path="final_evaluation.mp4", steps=1000):
 """
 Évaluer le modèle et créer une vidéo
 Basé sur le code du collègue
 """
 print("🎬 Création de la vidéo d'évaluation finale...")
 
 frames = []
 obs, _ = env.reset()
 
 total_reward = 0
 successful_grasps = 0
 
 for t in range(steps):
     action, _ = model.predict(obs, deterministic=True)
     obs, reward, terminated, truncated, info = env.step(action)
     
     total_reward += reward
     
     # Enregistrer la frame
     frame = env.render()
     if frame is not None:
         frames.append(Image.fromarray(frame.astype(np.uint8)))
     
     # Compter les succès
     if info.get('successful_grasp', False):
         successful_grasps += 1
     
     if terminated or truncated:
         obs, _ = env.reset()
 
 # Sauvegarder la vidéo à 30 fps comme le collègue
 if frames:
     imageio.mimsave(video_path, frames, fps=30)
     print(f"🎥 Vidéo d'évaluation sauvegardée: {video_path}")
     print(f"📊 Récompense totale: {total_reward:.2f}")
     print(f"🏆 Grasps réussis: {successful_grasps}")
 
 return total_reward, successful_grasps

def main():
 """Fonction principale"""
 print("🎯 ENTRAÎNEMENT TD3 POUR GRASPING ROBOTIQUE")
 print("=" * 60)
 
 # Entraînement
 model, env, eval_env = train_td3_robot()
 
 if model is not None:
     print("\n🎬 Création de la vidéo d'évaluation finale...")
     
     # Évaluation et vidéo finale
     video_path = "simple_td3_evaluation.mp4"
     total_reward, successes = evaluate_and_create_video(
         model, eval_env, video_path, steps=1000
     )
     
     print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
     print(f"📊 Performance finale: {total_reward:.2f} points")
     print(f"🏆 Succès: {successes}")
     print(f"🎥 Vidéo: {video_path}")
     
     # Test final de l'environnement
     print("\n🧪 Test final...")
     obs, info = env.reset()
     for i in range(5):
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, done, _, info = env.step(action)
         print(f"  Test step {i}: reward={reward:.3f}, distance={info.get('distance', 0):.3f}")
         if done:
             obs, info = env.reset()
 
 # Fermeture propre
 env.close()
 eval_env.close()
 print("✅ Environnements fermés proprement")

if __name__ == "__main__":
 main()
