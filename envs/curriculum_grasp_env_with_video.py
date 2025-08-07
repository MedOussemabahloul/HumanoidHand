
#!/usr/bin/env python3
"""
🎬 ENVIRONNEMENT DE GRASPING AVEC CURRICULUM LEARNING ET CAPTURE VIDÉO
=====================================================================

Version étendue avec capture vidéo pour visualiser:
🎥 Progression à travers les niveaux de curriculum
🎥 Évolution des mouvements du robot
🎥 Comparaison avant/après entraînement
🎥 Démonstration de la maîtrise de chaque niveau

Génère automatiquement des vidéos MP4 de l'entraînement !
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional
import tempfile
import warnings
import time
import imageio
warnings.filterwarnings("ignore")

# Import de l'environnement curriculum de base
import sys
sys.path.append('/workspace/envs')
from curriculum_grasp_env import CurriculumGraspEnv

class CurriculumGraspEnvWithVideo(CurriculumGraspEnv):
 """
 🎬 Environnement de Grasping avec Curriculum Learning et Capture Vidéo
 
 Étend l'environnement de base avec:
 - Capture vidéo automatique par niveau
 - Comparaisons avant/après
 - Démonstrations de maîtrise
 - Export MP4 haute qualité
 """
 
 def __init__(self, model_path: str = None, render_mode: str = "rgb_array", 
              video_folder: str = None, record_video: bool = True):
     
     # Configuration vidéo
     self.record_video = record_video
     self.video_folder = video_folder or "/workspace/curriculum_sac_results/videos"
     self.render_mode = render_mode if record_video else None
     
     # Initialiser l'environnement parent
     super().__init__(model_path, render_mode)
     
     # Variables de capture vidéo
     self.video_frames = []
     self.is_recording = False
     self.current_video_name = None
     self.episode_videos = {}
     self.level_demo_recorded = {}
     
     # Configuration de rendu
     if self.record_video:
         self._setup_video_capture()
     
     print(f"🎬 Environnement avec capture vidéo initialisé")
     if self.record_video:
         print(f"📁 Dossier vidéos: {self.video_folder}")
 
 def _setup_video_capture(self):
     """Configure la capture vidéo"""
     # Créer le dossier vidéos
     os.makedirs(self.video_folder, exist_ok=True)
     
     # Configuration de rendu MuJoCo pour vidéo
     self.video_width = 640
     self.video_height = 480
     self.video_fps = 30
     
     # Créer un contexte de rendu MuJoCo
     if hasattr(self, 'model') and hasattr(self, 'data'):
         try:
             # Créer le renderer
             self.renderer = mujoco.Renderer(self.model, self.video_height, self.video_width)
             print("✅ Renderer MuJoCo configuré pour vidéo")
         except Exception as e:
             print(f"⚠️ Erreur configuration renderer: {e}")
             self.record_video = False
 
 def start_video_recording(self, video_name: str):
     """Démarre l'enregistrement vidéo"""
     if not self.record_video:
         return
     
     self.current_video_name = video_name
     self.video_frames = []
     self.is_recording = True
     print(f"🎬 Début enregistrement: {video_name}")
 
 def stop_video_recording(self):
     """Arrête l'enregistrement et sauvegarde la vidéo"""
     if not self.record_video or not self.is_recording:
         return
     
     if len(self.video_frames) == 0:
         print("⚠️ Aucune frame enregistrée")
         return
     
     try:
         # Chemin de sauvegarde
         video_path = os.path.join(self.video_folder, f"{self.current_video_name}.mp4")
         
         # Sauvegarder avec imageio (plus robuste que OpenCV)
         with imageio.get_writer(video_path, fps=self.video_fps, quality=8) as writer:
             for frame in self.video_frames:
                 # Convertir BGR vers RGB si nécessaire
                 if len(frame.shape) == 3 and frame.shape[2] == 3:
                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     writer.append_data(frame_rgb)
                 else:
                     writer.append_data(frame)
         
         print(f"✅ Vidéo sauvée: {video_path} ({len(self.video_frames)} frames)")
         
         # Nettoyer
         self.video_frames = []
         self.is_recording = False
         
         return video_path
         
     except Exception as e:
         print(f"❌ Erreur sauvegarde vidéo: {e}")
         return None
 
 def capture_frame(self):
     """Capture une frame pour la vidéo"""
     if not self.record_video or not self.is_recording:
         return
     
     try:
         # Rendu MuJoCo
         if hasattr(self, 'renderer'):
             # Mettre à jour la vue avec les données actuelles
             self.renderer.update_scene(self.data)
             
             # Rendre l'image
             frame = self.renderer.render()
             
             # Ajouter des informations de debug sur l'image
             frame_with_info = self._add_info_overlay(frame)
             
             self.video_frames.append(frame_with_info)
             
     except Exception as e:
         print(f"⚠️ Erreur capture frame: {e}")
 
 def _add_info_overlay(self, frame):
     """Ajoute des informations de debug sur l'image"""
     try:
         # Convertir en format OpenCV
         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
         
         # Informations à afficher
         info_lines = [
             f"Niveau: {self.current_level} - {self.curriculum_levels[self.current_level]['name']}",
             f"Phase: {self._get_phase_name()} ({self.phase_timer})",
             f"Stabilite: {self.stability_count}",
             f"Episode: {getattr(self, 'episode_step', 0)}",
             f"Contacts: {self.contact_count}",
             f"Grasp: {'Oui' if self.successful_grasp else 'Non'}",
             f"Cube leve: {'Oui' if self.cube_lifted else 'Non'}"
         ]
         
         # Ajouter le texte sur l'image
         font = cv2.FONT_HERSHEY_SIMPLEX
         font_scale = 0.4
         color = (0, 255, 0)  # Vert
         thickness = 1
         
         y_offset = 20
         for i, line in enumerate(info_lines):
             y_pos = y_offset + i * 15
             cv2.putText(frame_bgr, line, (10, y_pos), font, font_scale, color, thickness)
         
         return frame_bgr
         
     except Exception as e:
         print(f"⚠️ Erreur overlay info: {e}")
         return frame
 
 def step(self, action):
     """Step avec capture vidéo"""
     # Exécuter le step normal
     obs, reward, terminated, truncated, info = super().step(action)
     
     # Capturer une frame
     self.capture_frame()
     
     return obs, reward, terminated, truncated, info
 
 def reset(self, seed=None, options=None):
     """Reset avec gestion vidéo"""
     # Reset normal
     obs, info = super().reset(seed, options)
     
     # Setup du renderer si pas encore fait
     if self.record_video and not hasattr(self, 'renderer'):
         self._setup_video_capture()
     
     # Capturer la frame initiale
     self.capture_frame()
     
     return obs, info
 
 def record_level_demonstration(self, level: int, num_episodes: int = 3):
     """Enregistre une démonstration d'un niveau spécifique"""
     if not self.record_video:
         return []
     
     print(f"🎬 Enregistrement démonstration niveau {level}")
     
     # Forcer le niveau
     original_level = self.current_level
     self.current_level = level
     self._update_phase_config()
     
     demo_videos = []
     
     for episode in range(num_episodes):
         video_name = f"demo_niveau_{level}_episode_{episode+1}"
         self.start_video_recording(video_name)
         
         # Épisode de démonstration
         obs, info = self.reset()
         episode_reward = 0
         
         for step in range(self.max_episode_steps):
             # Action aléatoire douce pour démonstration
             action = self.action_space.sample() * 0.05
             obs, reward, terminated, truncated, info = self.step(action)
             episode_reward += reward
             
             if terminated or truncated:
                 break
         
         # Arrêter l'enregistrement
         video_path = self.stop_video_recording()
         if video_path:
             demo_videos.append(video_path)
         
         print(f"  📹 Épisode {episode+1}: {step+1} steps, reward={episode_reward:.2f}")
     
     # Restaurer le niveau original
     self.current_level = original_level
     self._update_phase_config()
     
     return demo_videos
 
 def record_curriculum_progression(self, model, episodes_per_level: int = 2):
     """Enregistre la progression à travers tous les niveaux avec un modèle entraîné"""
     if not self.record_video:
         return {}
     
     print("🎬 Enregistrement progression curriculum complète")
     
     progression_videos = {}
     
     for level in range(1, 6):  # Niveaux 1 à 5
         print(f"📹 Niveau {level}: {self.curriculum_levels[level]['name']}")
         
         # Configurer le niveau
         self.current_level = level
         self._update_phase_config()
         
         level_videos = []
         
         for episode in range(episodes_per_level):
             video_name = f"progression_niveau_{level}_episode_{episode+1}"
             self.start_video_recording(video_name)
             
             # Épisode avec modèle entraîné
             obs, info = self.reset()
             episode_reward = 0
             
             for step in range(self.max_episode_steps):
                 if hasattr(model, 'predict'):
                     action, _ = model.predict(obs, deterministic=True)
                 else:
                     action = self.action_space.sample() * 0.1
                 
                 obs, reward, terminated, truncated, info = self.step(action)
                 episode_reward += reward
                 
                 if terminated or truncated:
                     break
             
             video_path = self.stop_video_recording()
             if video_path:
                 level_videos.append(video_path)
             
             print(f"  📹 Niveau {level} Épisode {episode+1}: {step+1} steps, reward={episode_reward:.2f}")
         
         progression_videos[level] = level_videos
     
     return progression_videos
 
 def create_comparison_video(self, before_model=None, after_model=None, level: int = 1):
     """Crée une vidéo de comparaison avant/après entraînement"""
     if not self.record_video:
         return None
     
     print(f"🎬 Création vidéo comparaison niveau {level}")
     
     # Configurer le niveau
     self.current_level = level
     self._update_phase_config()
     
     comparison_frames = []
     
     # Enregistrement "avant" (actions aléatoires ou modèle non entraîné)
     print("  📹 Séquence 'Avant entraînement'...")
     before_frames = []
     
     obs, info = self.reset()
     for step in range(200):  # 200 steps pour la démo
         if before_model and hasattr(before_model, 'predict'):
             action, _ = before_model.predict(obs, deterministic=True)
         else:
             action = self.action_space.sample() * 0.1
         
         obs, reward, terminated, truncated, info = self.step(action)
         
         # Capturer frame avec label "AVANT"
         if hasattr(self, 'renderer'):
             self.renderer.update_scene(self.data)
             frame = self.renderer.render()
             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
             
             # Ajouter label "AVANT"
             cv2.putText(frame_bgr, "AVANT ENTRAINEMENT", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
             
             frame_with_info = self._add_info_overlay(frame_bgr)
             before_frames.append(frame_with_info)
         
         if terminated or truncated:
             obs, info = self.reset()
     
     # Enregistrement "après" (modèle entraîné)
     print("  📹 Séquence 'Après entraînement'...")
     after_frames = []
     
     obs, info = self.reset()
     for step in range(200):
         if after_model and hasattr(after_model, 'predict'):
             action, _ = after_model.predict(obs, deterministic=True)
         else:
             action = self.action_space.sample() * 0.05  # Plus stable
         
         obs, reward, terminated, truncated, info = self.step(action)
         
         # Capturer frame avec label "APRÈS"
         if hasattr(self, 'renderer'):
             self.renderer.update_scene(self.data)
             frame = self.renderer.render()
             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
             
             # Ajouter label "APRÈS"
             cv2.putText(frame_bgr, "APRES ENTRAINEMENT", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
             
             frame_with_info = self._add_info_overlay(frame_bgr)
             after_frames.append(frame_with_info)
         
         if terminated or truncated:
             obs, info = self.reset()
     
     # Combiner les séquences côte à côte
     print("  🎞️ Création vidéo comparative...")
     try:
         video_path = os.path.join(self.video_folder, f"comparaison_niveau_{level}.mp4")
         
         with imageio.get_writer(video_path, fps=self.video_fps, quality=8) as writer:
             max_frames = min(len(before_frames), len(after_frames))
             
             for i in range(max_frames):
                 # Redimensionner si nécessaire
                 before_frame = cv2.resize(before_frames[i], (320, 240))
                 after_frame = cv2.resize(after_frames[i], (320, 240))
                 
                 # Combiner horizontalement
                 combined_frame = np.hstack([before_frame, after_frame])
                 
                 # Convertir pour imageio
                 combined_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                 writer.append_data(combined_rgb)
         
         print(f"✅ Vidéo comparaison sauvée: {video_path}")
         return video_path
         
     except Exception as e:
         print(f"❌ Erreur création vidéo comparaison: {e}")
         return None
 
 def close(self):
     """Fermeture avec nettoyage vidéo"""
     # Arrêter tout enregistrement en cours
     if self.is_recording:
         self.stop_video_recording()
     
     # Nettoyer le renderer
     if hasattr(self, 'renderer'):
         try:
             del self.renderer
         except:
             pass
     
     # Fermeture parent
     super().close()
 
 def get_video_summary(self):
     """Retourne un résumé des vidéos générées"""
     if not os.path.exists(self.video_folder):
         return {}
     
     videos = {}
     for file in os.listdir(self.video_folder):
         if file.endswith('.mp4'):
             file_path = os.path.join(self.video_folder, file)
             file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
             videos[file] = {
                 'path': file_path,
                 'size_mb': round(file_size, 2)
             }
     
     return videos

def create_demo_videos():
 """Fonction utilitaire pour créer des vidéos de démonstration"""
 print("🎬 GÉNÉRATION DE VIDÉOS DE DÉMONSTRATION")
 print("=" * 50)
 
 # Créer l'environnement avec vidéo
 env = CurriculumGraspEnvWithVideo(record_video=True)
 
 try:
     # Démonstrations par niveau
     for level in range(1, 4):  # Premiers 3 niveaux
         env.record_level_demonstration(level, num_episodes=2)
     
     # Résumé des vidéos créées
     videos = env.get_video_summary()
     print(f"\n📹 Vidéos créées: {len(videos)}")
     for name, info in videos.items():
         print(f"  📹 {name}: {info['size_mb']} MB")
     
     return videos
     
 finally:
     env.close()

if __name__ == "__main__":
 # Test de création de vidéos
 create_demo_videos()
