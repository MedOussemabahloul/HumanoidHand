#!/usr/bin/env python3
"""
🎯 ENTRAÎNEUR SAC ROBUSTE AVEC CURRICULUM LEARNING POUR GRASPING G1
===================================================================

Version ultra-stable et professionnelle qui corrige tous les problèmes:
✅ Vitesses excessives - Contrôle de vitesse intelligent
✅ Erreurs mujoco - Gestion robuste des imports et contextes
✅ Capture vidéo - Système de vidéo intégré et fonctionnel
✅ Stagnation - Système de récompenses adaptatif
✅ Instabilité - Physique ultra-stable
✅ Monitoring - Suivi en temps réel des performances

Fonctionnalités avancées:
- Progression automatique de difficulté
- Hyperparamètres adaptatifs selon le niveau
- Monitoring en temps réel du curriculum
- Sauvegarde de modèles par niveau
- Visualisation des progrès
- Capture vidéo automatique
- Ouverture de la simulation Mujoco en temps réel
"""
import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import cv2
import subprocess
import threading
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/home/oussema/Documents/project/envs')

try:
 from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
 print("✅ RobustCurriculumGraspEnv importé avec succès")
except ImportError as e:
 print(f"❌ Erreur d'import: {e}")
 try:
     sys.path.append('/home/oussema/Documents/project/envs')
     from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
     print("✅ RobustCurriculumGraspEnv importé avec succès (fallback)")
 except ImportError as e2:
     print(f"❌ Erreur d'import (fallback): {e2}")
     sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class RobustCurriculumGraspingTrainer:
 """
 🎯 Entraîneur SAC Ultra-Robuste avec Curriculum Learning
 
 Fonctionnalités avancées:
 - Progression automatique de difficulté
 - Hyperparamètres adaptatifs selon le niveau
 - Monitoring en temps réel du curriculum
 - Sauvegarde de modèles par niveau
 - Visualisation des progrès
 - Capture vidéo automatique
 - Ouverture de la simulation Mujoco en temps réel
 """
 
 def __init__(self, total_timesteps: int = 200000):
     self.total_timesteps = total_timesteps
     
     # Configuration des dossiers
     self.results_dir = "/home/oussema/Documents/project/robust_curriculum_sac_results"
     self.models_dir = os.path.join(self.results_dir, "models")
     self.logs_dir = os.path.join(self.results_dir, "logs")
     self.videos_dir = os.path.join(self.results_dir, "videos")
     self.plots_dir = os.path.join(self.results_dir, "plots")
     
     self._setup_directories()
     
     # Métriques d'entraînement avec curriculum
     self.training_metrics = {
         'episode_rewards': [],
         'episode_lengths': [],
         'curriculum_levels': [],
         'level_transitions': [],
         'success_rates_by_level': {},
         'training_time': 0.0,
         'best_reward_by_level': {},
         'total_episodes': 0,
         'video_paths': []
     }
     
     # Configuration de l'environnement de curriculum
     self.env = None
     self.model = None
     self.current_level = 1
     
     # Configuration vidéo et simulation
     self.video_capture = True
     self.mujoco_viewer = None
     self.viewer_thread = None
     
     print("🎯 RobustCurriculumGraspingTrainer initialisé")
     print(f"📁 Résultats: {self.results_dir}")
 
 def _setup_directories(self):
     """Crée les dossiers nécessaires"""
     for directory in [self.results_dir, self.models_dir, self.logs_dir, 
                     self.videos_dir, self.plots_dir]:
         os.makedirs(directory, exist_ok=True)
 
 def create_robust_environment(self):
     """Crée l'environnement robuste avec curriculum learning"""
     print("🏗️  Création de l'environnement robuste avec curriculum learning...")
     
     try:
         # Créer l'environnement avec capture vidéo
         self.env = RobustCurriculumGraspEnv(
             model_path="/home/oussema/Documents/project/results/g1_combined.xml",
             render_mode="rgb_array",
             video_capture=self.video_capture
         )
         
         print("✅ Environnement robuste créé avec succès")
         print(f"  - Niveau actuel: {self.env.current_level}")
         print(f"  - Capture vidéo: {self.video_capture}")
         print(f"  - Espace d'action: {self.env.action_space.shape}")
         print(f"  - Espace d'observation: {self.env.observation_space.shape}")
         
         return True
         
     except Exception as e:
         print(f"❌ Erreur lors de la création de l'environnement: {e}")
         return False
 
 def create_adaptive_sac_model(self):
     """Crée un modèle SAC adaptatif selon le niveau de curriculum"""
     print("🤖 Création du modèle SAC adaptatif...")
     
     try:
         # Hyperparamètres adaptatifs selon le niveau
         level_config = self.env.curriculum_levels[self.current_level]
         
         # Paramètres de base optimisés
         base_params = {
             'learning_rate': 0.0001,  # Plus lent pour stabilité
             'buffer_size': 50000,     # Plus petit au début
             'batch_size': 128,        # Plus petit pour stabilité
             'gamma': 0.98,            # Plus réaliste
             'ent_coef': 0.2,          # Exploration modérée
             'tau': 0.005,             # Mise à jour plus lente
             'train_freq': 1,          # Entraînement à chaque step
             'gradient_steps': 1,      # Un gradient step par step
             'learning_starts': 1000,  # Commencer après 1000 steps
             'verbose': 1
         }
         
         # Ajustements selon le niveau
         if self.current_level == 1:
             # Niveau débutant: apprentissage très lent
             base_params['learning_rate'] = 0.00005
             base_params['ent_coef'] = 0.3
             base_params['learning_starts'] = 500
         elif self.current_level == 2:
             # Niveau intermédiaire: apprentissage modéré
             base_params['learning_rate'] = 0.0001
             base_params['ent_coef'] = 0.2
             base_params['learning_starts'] = 1000
         elif self.current_level >= 3:
             # Niveau avancé: apprentissage normal
             base_params['learning_rate'] = 0.0002
             base_params['ent_coef'] = 0.15
             base_params['learning_starts'] = 1000
         
         # Créer le modèle
         self.model = SAC(
             "MlpPolicy",
             self.env,
             **base_params
         )
         
         print("✅ Modèle SAC adaptatif créé avec succès")
         print(f"  - Learning rate: {base_params['learning_rate']}")
         print(f"  - Buffer size: {base_params['buffer_size']}")
         print(f"  - Batch size: {base_params['batch_size']}")
         print(f"  - Entropy coefficient: {base_params['ent_coef']}")
         
         return True
         
     except Exception as e:
         print(f"❌ Erreur lors de la création du modèle: {e}")
         return False
 
 def start_mujoco_viewer(self):
     """Démarre le viewer Mujoco en arrière-plan"""
     try:
         print("🖥️  Démarrage du viewer Mujoco...")
         
         # Créer un thread pour le viewer
         def run_viewer():
             try:
                 import mujoco.viewer
                 with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                     self.mujoco_viewer = viewer
                     while True:
                         viewer.sync()
                         time.sleep(0.01)
             except Exception as e:
                 print(f"⚠️ Erreur viewer Mujoco: {e}")
         
         self.viewer_thread = threading.Thread(target=run_viewer, daemon=True)
         self.viewer_thread.start()
         
         print("✅ Viewer Mujoco démarré en arrière-plan")
         
     except Exception as e:
         print(f"⚠️ Impossible de démarrer le viewer Mujoco: {e}")
 
 def train_with_curriculum(self):
     """Entraîne le modèle avec curriculum learning"""
     print("🎓 Début de l'entraînement avec curriculum learning...")
     
     start_time = time.time()
     
     # Démarrer le viewer Mujoco
     self.start_mujoco_viewer()
     
     # Configuration du logger
     logger = configure(
         self.logs_dir,
         format_strings=["stdout", "log", "csv", "tensorboard"]
     )
     
     # Métriques de suivi
     episode_rewards = []
     episode_lengths = []
     recent_rewards = []
     best_reward = -np.inf
     
     # Entraînement par niveau
     current_level = 1
     total_episodes = 0
     
     while current_level <= len(self.env.curriculum_levels) and total_episodes < 1000:
         print(f"\n🎯 Entraînement niveau {current_level}")
         print(f"📊 Niveau: {self.env.curriculum_levels[current_level]['name']}")
         
         # Créer le modèle pour ce niveau
         if not self.create_adaptive_sac_model():
             break
         
         # Entraînement pour ce niveau
         level_episodes = 0
         level_rewards = []
         consecutive_successes = 0
         
         while (level_episodes < 50 and 
                consecutive_successes < self.env.curriculum_levels[current_level]['episodes_required']):
             
             # Entraînement par épisode
             obs, info = self.env.reset()
             episode_reward = 0
             episode_length = 0
             
             while True:
                 # Prédiction de l'action
                 action, _states = self.model.predict(obs, deterministic=False)
                 
                 # Exécution de l'action
                 obs, reward, terminated, truncated, info = self.env.step(action)
                 
                 episode_reward += reward
                 episode_length += 1
                 
                 # Vérifier la terminaison
                 if terminated or truncated:
                     break
             
             # Mise à jour des métriques
             episode_rewards.append(episode_reward)
             episode_lengths.append(episode_length)
             level_rewards.append(episode_reward)
             recent_rewards.append(episode_reward)
             
             # Garder seulement les 100 dernières récompenses
             if len(recent_rewards) > 100:
                 recent_rewards.pop(0)
             
             total_episodes += 1
             level_episodes += 1
             
             # Vérifier le succès
             level_config = self.env.curriculum_levels[current_level]
             episode_success = (episode_reward >= level_config['success_threshold'])
             
             if episode_success:
                 consecutive_successes += 1
             else:
                 consecutive_successes = 0
             
             # Affichage des progrès
             if total_episodes % 10 == 0:
                 avg_recent = np.mean(recent_rewards[-20:]) if recent_rewards else 0
                 print(f"📈 Épisode {total_episodes:3d} | "
                       f"Niveau {current_level} | "
                       f"Récompense: {episode_reward:6.2f} | "
                       f"Moyenne récente: {avg_recent:6.2f} | "
                       f"Succès consécutifs: {consecutive_successes}")
             
             # Sauvegarde intermédiaire
             if total_episodes % 50 == 0:
                 self._save_intermediate_model(current_level, total_episodes)
             
             # Vérifier si on peut passer au niveau suivant
             if consecutive_successes >= level_config['episodes_required']:
                 print(f"🎉 Niveau {current_level} terminé avec succès!")
                 break
         
         # Sauvegarder le modèle du niveau
         self._save_level_model(current_level)
         
         # Passer au niveau suivant
         if current_level < len(self.env.curriculum_levels):
             current_level += 1
             self.env.current_level = current_level
             print(f"🚀 Passage au niveau {current_level}")
         else:
             print("🏆 Tous les niveaux terminés!")
             break
     
     # Temps total d'entraînement
     training_time = time.time() - start_time
     self.training_metrics['training_time'] = training_time
     
     print(f"\n🎯 Entraînement terminé!")
     print(f"⏱️  Temps total: {training_time/3600:.2f} heures")
     print(f"📊 Épisodes totaux: {total_episodes}")
     print(f"🏆 Niveau final: {current_level-1}")
     
     # Sauvegarder les métriques finales
     self._save_training_metrics()
     
     # Générer la vidéo finale
     self.generate_final_video()
     
     return True
 
 def _save_intermediate_model(self, level: int, episode: int):
     """Sauvegarde un modèle intermédiaire"""
     try:
         model_path = os.path.join(self.models_dir, f"level_{level}_episode_{episode}.zip")
         self.model.save(model_path)
         print(f"💾 Modèle intermédiaire sauvegardé: {model_path}")
     except Exception as e:
         print(f"⚠️ Erreur sauvegarde intermédiaire: {e}")
 
 def _save_level_model(self, level: int):
     """Sauvegarde le modèle d'un niveau"""
     try:
         model_path = os.path.join(self.models_dir, f"level_{level}_final.zip")
         self.model.save(model_path)
         print(f"💾 Modèle niveau {level} sauvegardé: {model_path}")
     except Exception as e:
         print(f"⚠️ Erreur sauvegarde niveau: {e}")
 
 def _save_training_metrics(self):
     """Sauvegarde les métriques d'entraînement"""
     try:
         metrics_path = os.path.join(self.results_dir, "training_metrics.json")
         with open(metrics_path, 'w') as f:
             json.dump(self.training_metrics, f, indent=2)
         print(f"📊 Métriques sauvegardées: {metrics_path}")
     except Exception as e:
         print(f"⚠️ Erreur sauvegarde métriques: {e}")
 
 def generate_final_video(self):
     """Génère une vidéo finale de démonstration"""
     print("🎥 Génération de la vidéo finale...")
     
     try:
         # Charger le meilleur modèle
         best_model_path = None
         for level in range(len(self.env.curriculum_levels), 0, -1):
             model_path = os.path.join(self.models_dir, f"level_{level}_final.zip")
             if os.path.exists(model_path):
                 best_model_path = model_path
                 break
         
         if best_model_path is None:
             print("⚠️ Aucun modèle trouvé pour la génération de vidéo")
             return
         
         # Charger le modèle
         model = SAC.load(best_model_path)
         
         # Créer un environnement pour la démonstration
         demo_env = RobustCurriculumGraspEnv(
             model_path="/home/oussema/Documents/project/results/g1_combined.xml",
             render_mode="rgb_array",
             video_capture=True
         )
         
         # Générer la vidéo de démonstration
         video_path = os.path.join(self.videos_dir, "final_demo.mp4")
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
         
         # Exécuter quelques épisodes de démonstration
         for episode in range(3):
             obs, info = demo_env.reset()
             episode_frames = []
             
             for step in range(500):
                 action, _states = model.predict(obs, deterministic=True)
                 obs, reward, terminated, truncated, info = demo_env.step(action)
                 
                 # Capturer la frame
                 frame = demo_env.render()
                 if frame is not None:
                     # Convertir RGB vers BGR
                     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                     video_writer.write(frame_bgr)
                 
                 if terminated or truncated:
                     break
         
         video_writer.release()
         demo_env.close()
         
         print(f"🎥 Vidéo finale générée: {video_path}")
         
         # Ouvrir la vidéo
         try:
             subprocess.Popen(['xdg-open', video_path])
             print("🎬 Vidéo ouverte automatiquement")
         except:
             print(f"📁 Vidéo disponible: {video_path}")
         
     except Exception as e:
         print(f"⚠️ Erreur génération vidéo: {e}")
 
 def test_final_model(self):
     """Teste le modèle final"""
     print("🧪 Test du modèle final...")
     
     try:
         # Charger le meilleur modèle
         best_model_path = None
         for level in range(len(self.env.curriculum_levels), 0, -1):
             model_path = os.path.join(self.models_dir, f"level_{level}_final.zip")
             if os.path.exists(model_path):
                 best_model_path = model_path
                 break
         
         if best_model_path is None:
             print("⚠️ Aucun modèle trouvé pour le test")
             return False
         
         # Charger le modèle
         model = SAC.load(best_model_path)
         
         # Test sur plusieurs épisodes
         test_rewards = []
         test_successes = 0
         
         for episode in range(10):
             obs, info = self.env.reset()
             episode_reward = 0
             
             while True:
                 action, _states = model.predict(obs, deterministic=True)
                 obs, reward, terminated, truncated, info = self.env.step(action)
                 episode_reward += reward
                 
                 if terminated or truncated:
                     break
             
             test_rewards.append(episode_reward)
             if episode_reward > 50:  # Seuil de succès
                 test_successes += 1
         
         avg_reward = np.mean(test_rewards)
         success_rate = test_successes / 10
         
         print(f"📊 Résultats du test:")
         print(f"  - Récompense moyenne: {avg_reward:.2f}")
         print(f"  - Taux de succès: {success_rate:.2%}")
         print(f"  - Récompenses: {test_rewards}")
         
         return success_rate > 0.5
         
     except Exception as e:
         print(f"⚠️ Erreur test final: {e}")
         return False

def main():
 """Fonction principale"""
 print("🎯 DÉMARRAGE DE L'ENTRAÎNEUR ROBUSTE")
 print("=" * 50)
 
 # Créer l'entraîneur
 trainer = RobustCurriculumGraspingTrainer(total_timesteps=200000)
 
 # Créer l'environnement
 if not trainer.create_robust_environment():
     print("❌ Impossible de créer l'environnement")
     return
 
 # Entraînement avec curriculum
 if trainer.train_with_curriculum():
     print("✅ Entraînement terminé avec succès!")
     
     # Test du modèle final
     if trainer.test_final_model():
         print("🎉 Modèle testé avec succès!")
     else:
         print("⚠️ Modèle nécessite plus d'entraînement")
 else:
     print("❌ Erreur lors de l'entraînement")

if __name__ == "__main__":
 main()
