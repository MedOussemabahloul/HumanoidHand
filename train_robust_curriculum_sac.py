#!/usr/bin/env python3
"""
🎓 ENTRAÎNEUR SAC ROBUSTE AVEC CURRICULUM LEARNING
=================================================

Version optimisée et robuste du système de grasping avec curriculum learning.
Basé sur les tests réussis, ce script garantit un fonctionnement stable.
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import warnings
import cv2
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/home/oussema/Documents/project/envs')

try:
  from envs.curriculum_grasp_env import CurriculumGraspEnv
  print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
  print(f"❌ Erreur d'import: {e}")
  sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumProgressCallback(BaseCallback):
  """Callback pour suivre les progrès du curriculum learning"""
  
  def __init__(self, check_freq: int = 1000, verbose=0):
      super().__init__(verbose)
      self.check_freq = check_freq
      self.episode_rewards = []
      self.episode_count = 0
      
  def _on_step(self) -> bool:
      # Enregistrer les récompenses d'épisode
      if self.locals.get('dones', [False])[0]:
          if 'episode' in self.locals.get('infos', [{}])[0]:
              episode_reward = self.locals['infos'][0]['episode']['r']
              self.episode_rewards.append(episode_reward)
              self.episode_count += 1
              
              # Log des progrès curriculum
              if hasattr(self.training_env.envs[0], 'get_curriculum_info'):
                  curriculum_info = self.training_env.envs[0].get_curriculum_info()
                  self.logger.record("curriculum/level", curriculum_info['current_level'])
                  self.logger.record("curriculum/consecutive_successes", curriculum_info['consecutive_successes'])
                  self.logger.record("curriculum/level_episodes", curriculum_info['level_episodes'])
              
              # Afficher progrès périodiquement
              if self.episode_count % 10 == 0:
                  recent_rewards = self.episode_rewards[-10:]
                  avg_reward = np.mean(recent_rewards)
                  print(f"📊 Épisode {self.episode_count}: Récompense moyenne (10 derniers): {avg_reward:.2f}")
      
      return True

class RobustCurriculumTrainer:
  """Entraîneur robuste avec curriculum learning et capture vidéo"""
  
  def __init__(self, total_timesteps: int = 50000):
      self.total_timesteps = total_timesteps
      
      # Configuration des dossiers
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      self.results_dir = f"/home/oussema/Documents/project/curriculum_sac_results_{timestamp}"
      self.models_dir = os.path.join(self.results_dir, "models")
      self.logs_dir = os.path.join(self.results_dir, "logs")
      self.videos_dir = os.path.join(self.results_dir, "videos")
      
      # Créer les dossiers
      for directory in [self.results_dir, self.models_dir, self.logs_dir, self.videos_dir]:
          os.makedirs(directory, exist_ok=True)
      
      print(f"🎓 RobustCurriculumTrainer initialisé")
      print(f"📁 Résultats: {self.results_dir}")
      
      # Métriques d'entraînement
      self.training_metrics = {
          'start_time': time.time(),
          'total_episodes': 0,
          'training_time': 0,
          'success_rates_by_level': {},
          'best_reward_by_level': {},
          'level_transitions': [],
          'final_level': 1
      }
      
  def create_environment(self):
      """Créer l'environnement avec gestion d'erreurs robuste"""
      try:
          print("🏗️  Création de l'environnement curriculum...")
          self.env = CurriculumGraspEnv()
          print("✅ Environnement créé avec succès")
          return True
      except Exception as e:
          print(f"❌ Erreur création environnement: {e}")
          return False
  
  def create_model(self):
      """Créer le modèle SAC avec configuration optimisée"""
      try:
          print("🧠 Création du modèle SAC robuste...")
          
          self.model = SAC(
              "MlpPolicy",
              self.env,
              learning_rate=0.0003,  # Taux d'apprentissage stable
              buffer_size=100000,    # Buffer plus grand pour meilleure stabilité
              batch_size=256,        # Batch size optimisé
              tau=0.005,            # Soft update coefficient
              gamma=0.99,           # Discount factor
              train_freq=1,         # Entraîner à chaque step
              gradient_steps=1,     # Gradient steps par update
              verbose=1,
              device="auto",        # Auto-détection GPU/CPU
              tensorboard_log=self.logs_dir
          )
          
          # Configuration du logger
          logger = configure(self.logs_dir, ["stdout", "csv", "tensorboard"])
          self.model.set_logger(logger)
          
          print("✅ Modèle SAC robuste créé")
          print(f"  - Learning rate: {self.model.learning_rate}")
          print(f"  - Buffer size: {self.model.buffer_size}")
          print(f"  - Batch size: {self.model.batch_size}")
          print(f"  - Device: {self.model.device}")
          
          return True
          
      except Exception as e:
          print(f"❌ Erreur création modèle: {e}")
          return False
  
  def train_with_monitoring(self):
      """Entraînement avec monitoring avancé"""
      try:
          print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT ROBUSTE")
          print("=" * 50)
          
          # Callback pour monitoring
          callback = CurriculumProgressCallback(check_freq=1000)
          
          # Entraînement principal
          print(f"📚 Entraînement pour {self.total_timesteps} timesteps...")
          start_time = time.time()
          
          self.model.learn(
              total_timesteps=self.total_timesteps,
              callback=callback,
              log_interval=10,
              reset_num_timesteps=False
          )
          
          training_time = time.time() - start_time
          self.training_metrics['training_time'] = training_time
          self.training_metrics['total_episodes'] = callback.episode_count
          
          print(f"✅ Entraînement terminé en {training_time:.2f}s")
          print(f"📊 Total d'épisodes: {callback.episode_count}")
          
          # Sauvegarder le modèle final
          model_path = os.path.join(self.models_dir, "robust_curriculum_sac_final.zip")
          self.model.save(model_path)
          print(f"💾 Modèle final sauvé: {model_path}")
          
          # Obtenir les métriques finales du curriculum
          if hasattr(self.env, 'get_curriculum_info'):
              curriculum_info = self.env.get_curriculum_info()
              self.training_metrics['final_level'] = curriculum_info['current_level']
              print(f"🎓 Niveau final atteint: {curriculum_info['current_level']}")
          
          return True
          
      except Exception as e:
          print(f"❌ Erreur durant l'entraînement: {e}")
          import traceback
          traceback.print_exc()
          return False
  
  def generate_demo_video(self, num_episodes: int = 3, max_steps_per_episode: int = 500):
      """Générer une vidéo de démonstration robuste"""
      try:
          print("\n🎬 GÉNÉRATION DE LA VIDÉO DE DÉMONSTRATION")
          print("=" * 50)
          
          # Créer environnement pour vidéo
          video_env = CurriculumGraspEnv(render_mode='rgb_array')
          
          # Définir le niveau au maximum atteint
          if hasattr(self.env, 'current_level'):
              video_env.current_level = self.env.current_level
              video_env._update_curriculum_config()
          
          # Configuration vidéo
          video_path = os.path.join(self.videos_dir, "demonstration.mp4")
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          fps = 30
          frame_size = (640, 480)
          
          print(f"📹 Enregistrement: {video_path}")
          
          video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
          if not video_writer.isOpened():
              print("⚠️ OpenCV mp4v non disponible, tentative avec 'avc1'")
              fallback_fourcc = cv2.VideoWriter_fourcc(*'avc1')
              video_writer = cv2.VideoWriter(video_path, fallback_fourcc, fps, frame_size)
          if not video_writer.isOpened():
              raise RuntimeError(f"Impossible d'ouvrir l'écrivain vidéo: {video_path}")
          total_frames = 0
          successful_episodes = 0
          
          for episode in range(num_episodes):
              print(f"🎬 Épisode {episode + 1}/{num_episodes}")
              
              obs, info = video_env.reset()
              episode_reward = 0
              episode_success = False
              
              for step in range(max_steps_per_episode):
                  # Prédiction déterministe
                  action, _ = self.model.predict(obs, deterministic=True)
                  obs, reward, terminated, truncated, info = video_env.step(action)
                  episode_reward += reward
                  
                  # Capture de frame
                  try:
                      frame = video_env.render()
                      if frame is not None and frame.size > 0:
                          # Redimensionner si nécessaire
                          if frame.shape[:2] != frame_size[::-1]:
                              frame = cv2.resize(frame, frame_size)
                          
                          # Convertir RGB vers BGR pour OpenCV
                          if len(frame.shape) == 3 and frame.shape[2] == 3:
                              frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                              video_writer.write(frame)
                              total_frames += 1
                  except Exception as frame_error:
                      print(f"⚠️  Erreur capture frame: {frame_error}")
                  
                  if terminated or truncated:
                      if info.get('successful_grasp', False) or reward > 10:
                          episode_success = True
                          successful_episodes += 1
                      break
              
              print(f"   Récompense: {episode_reward:.2f}, Succès: {episode_success}")
          
          video_writer.release()
          video_env.close()
          
          # Vérifier la vidéo créée
          if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
              print(f"✅ Vidéo générée avec succès!")
              print(f"   📁 Chemin: {video_path}")
              print(f"   🎬 Frames: {total_frames}")
              print(f"   ✨ Épisodes réussis: {successful_episodes}/{num_episodes}")
              
              # Créer GIF (optionnel)
              self._create_gif_from_video(video_path)
              return True
          else:
              print("❌ Erreur: Vidéo non créée ou vide")
              return False
              
      except Exception as e:
          print(f"❌ Erreur génération vidéo: {e}")
          import traceback
          traceback.print_exc()
          return False
  
  def _create_gif_from_video(self, video_path: str):
      """Créer un GIF à partir de la vidéo"""
      try:
          gif_path = video_path.replace('.mp4', '.gif')
          
          cap = cv2.VideoCapture(video_path)
          if not cap.isOpened():
              raise RuntimeError(f"Impossible d'ouvrir la vidéo pour lecture: {video_path}")
          frames = []
          frame_count = 0
          
          while True:
              ret, frame = cap.read()
              if not ret:
                  break
              
              # Prendre une frame sur 3 pour réduire la taille
              if frame_count % 3 == 0:
                  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  frame_small = cv2.resize(frame_rgb, (320, 240))
                  frames.append(frame_small)
              
              frame_count += 1
          
          cap.release()
          
          if frames:
              from PIL import Image
              pil_frames = [Image.fromarray(frame) for frame in frames]
              pil_frames[0].save(
                  gif_path,
                  save_all=True,
                  append_images=pil_frames[1:],
                  duration=100,
                  loop=0
              )
              print(f"🎞️  GIF créé: {gif_path}")
          
      except Exception as e:
          print(f"⚠️  Impossible de créer le GIF: {e}")
  
  def save_metrics(self):
      """Sauvegarder les métriques d'entraînement"""
      try:
          metrics_path = os.path.join(self.results_dir, "training_metrics.json")
          with open(metrics_path, 'w') as f:
              json.dump(self.training_metrics, f, indent=2)
          
          # Résumé lisible
          summary_path = os.path.join(self.results_dir, "training_summary.txt")
          with open(summary_path, 'w') as f:
              f.write("🎓 RÉSUMÉ DE L'ENTRAÎNEMENT ROBUSTE\n")
              f.write("=" * 50 + "\n\n")
              f.write(f"Niveau final: {self.training_metrics['final_level']}\n")
              f.write(f"Épisodes totaux: {self.training_metrics['total_episodes']}\n")
              f.write(f"Temps d'entraînement: {self.training_metrics['training_time']:.2f}s\n")
              f.write(f"Timesteps: {self.total_timesteps}\n")
          
          print(f"📊 Métriques sauvées: {metrics_path}")
          print(f"📄 Résumé sauvé: {summary_path}")
          
      except Exception as e:
          print(f"⚠️  Erreur sauvegarde métriques: {e}")
  
  def cleanup(self):
      """Nettoyage des ressources"""
      if hasattr(self, 'env'):
          self.env.close()
          print("🧹 Environnement fermé")

def main():
  """Fonction principale d'entraînement robuste"""
  print("🎓 LANCEMENT DE L'ENTRAÎNEMENT SAC ROBUSTE")
  print("=" * 60)
  
  # Configuration
  total_timesteps = 50000  # Entraînement modéré pour test
  
  trainer = RobustCurriculumTrainer(total_timesteps=total_timesteps)
  
  try:
      # 1. Créer l'environnement
      if not trainer.create_environment():
          print("❌ Échec création environnement")
          return
      
      # 2. Créer le modèle
      if not trainer.create_model():
          print("❌ Échec création modèle")
          return
      
      # 3. Entraîner avec monitoring
      if not trainer.train_with_monitoring():
          print("❌ Échec entraînement")
          return
      
      # 4. Générer vidéo de démonstration
      print("\n🎬 Génération de la vidéo...")
      trainer.generate_demo_video()
      
      # 5. Sauvegarder métriques
      trainer.save_metrics()
      
      print("\n🎉 ENTRAÎNEMENT ROBUSTE COMPLÉTÉ AVEC SUCCÈS!")
      print(f"📁 Tous les résultats dans: {trainer.results_dir}")
      
  except KeyboardInterrupt:
      print("\n⏹️  Entraînement interrompu par l'utilisateur")
      
  except Exception as e:
      print(f"\n❌ Erreur fatale: {e}")
      import traceback
      traceback.print_exc()
      
  finally:
      trainer.cleanup()

if __name__ == "__main__":
  main()
