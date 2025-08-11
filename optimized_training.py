#!/usr/bin/env python3
"""
🚀 TRAINING OPTIMISÉ - INSPIRÉ DU COLLÈGUE
==========================================

Script de training qui combine les meilleures pratiques du collègue
avec notre approche professionnelle:

✅ INSPIRATIONS DU COLLÈGUE:
-TD3 avec hyperparamètres qui fonctionnent
- Évaluation avec vidéos intégrées
- Approche simple et directe

✅ NOTRE VALEUR AJOUTÉE:
- Curriculum learning progressif
- Gestion robuste des erreurs
- Monitoring avancé des performances
- Sauvegarde automatique des meilleurs modèles
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import logging

# Configuration des warnings
warnings.filterwarnings("ignore")

# Imports ML
try:
 import torch
 from stable_baselines3 import TD3, SAC
 from stable_baselines3.common.noise import NormalActionNoise
 from stable_baselines3.common.callbacks import BaseCallback
 from stable_baselines3.common.monitor import Monitor
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

# Import de notre environnement optimisé
try:
 from envs.optimized_grasp_env import OptimizedGraspEnv, make_optimized_grasp_env
 print("✅ Environnement optimisé importé")
except ImportError:
 print("❌ ERREUR: Impossible d'importer l'environnement optimisé")
 sys.exit(1)


class OptimizedTrainingCallback(BaseCallback):
 """
 Callback optimisé inspiré du collègue avec nos améliorations
 """
 
 def __init__(self, 
             eval_freq: int = 25000,
             video_freq: int = 50000,
             save_freq: int = 25000,
             n_eval_episodes: int = 5,
             results_dir: str = "optimized_results",
             verbose: int = 1):
     
     super().__init__(verbose)
     
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
     
     # Métriques
     self.episode_rewards = []
     self.eval_rewards = []
     self.best_eval_reward = -np.inf
     self.curriculum_history = []
     
     # Détection de stagnation
     self.stagnation_counter = 0
     self.last_improvement_step = 0
     self.stagnation_threshold = 50000  # Redémarrer si pas d'amélioration en 50k steps
     
     # Environnement d'évaluation
     self.eval_env = None
     
     # Logger
     self.setup_logging()
 
 def setup_logging(self):
     """Configure le logging"""
     self.custom_logger = logging.getLogger("OptimizedTraining")
     self.custom_logger.setLevel(logging.INFO)
     
     if not self.custom_logger.handlers:
         # Handler fichier
         handler = logging.FileHandler(self.results_dir / "training.log")
         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
         self.eval_env = OptimizedGraspEnv(render_mode="rgb_array")
         self.custom_logger.info("✅ Environnement d'évaluation créé")
     except Exception as e:
         self.custom_logger.error(f"❌ Erreur création env d'évaluation: {e}")
 
 def _on_step(self) -> bool:
     """Appelé à chaque step"""
     
     # Logging périodique avec métriques dans TensorBoard
     if self.n_calls % 500 == 0:  # Plus fréquent pour voir la progression
         try:
             infos = self.locals.get('infos', [{}])
             if infos and len(infos) > 0:
                 info = infos[0]
                 
                 # Extraire métriques
                 dist = info.get('distance', 0)
                 contacts = info.get('contact_count', 0)
                 curriculum = info.get('curriculum_level', 1)
                 
                 # Affichage complet avec toutes les métriques
                 reward = info.get('episode_reward', 0)
                 print(f"📊 Step {self.n_calls}: DISTANCE={dist:.3f}m, contacts={contacts}, reward={reward:.1f}, curriculum={curriculum}")
                 
                 # AJOUTER LES MÉTRIQUES À TENSORBOARD/LOGS SB3
                 if hasattr(self, 'logger') and self.logger:
                     self.logger.record("train/distance", dist)
                     self.logger.record("train/contacts", contacts)
                     self.logger.record("train/curriculum_level", curriculum)
                     self.logger.record("train/episode_reward", reward)
                 
         except Exception as e:
             self.custom_logger.warning(f"Erreur logging: {e}")
     
     # Évaluation périodique
     if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
         self._run_evaluation()
     
     # Vidéo périodique
     if self.n_calls % self.video_freq == 0 and self.n_calls > 0:
         self._create_video()
     
     # Sauvegarde périodique
     if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
         self._save_model()
     
     return True
 
 def _run_evaluation(self):
     """Évaluation comme le collègue mais avec curriculum"""
     
     if not self.eval_env:
         return
     
     self.custom_logger.info(f"📊 Évaluation à {self.n_calls} steps...")
     
     eval_rewards = []
     eval_contacts = []
     eval_distances = []
     
     for episode in range(self.n_eval_episodes):
         obs, _ = self.eval_env.reset()
         episode_reward = 0
         episode_contacts = 0
         final_distance = float('inf')
         
         for step in range(500):
             action, _ = self.model.predict(obs, deterministic=True)
             obs, reward, terminated, _, info = self.eval_env.step(action)
             
             episode_reward += reward
             episode_contacts = max(episode_contacts, info.get('contact_count', 0))
             final_distance = info.get('distance', final_distance)
             
             if terminated:
                 break
         
         eval_rewards.append(episode_reward)
         eval_contacts.append(episode_contacts)
         eval_distances.append(final_distance)
     
     # Statistiques
     mean_reward = np.mean(eval_rewards)
     mean_contacts = np.mean(eval_contacts)
     mean_distance = np.mean(eval_distances)
     
     # Enregistrer dans les logs
     self.eval_rewards.append(mean_reward)
     
     # Sauvegarder le meilleur modèle et détecter stagnation
     if mean_reward > self.best_eval_reward:
         self.best_eval_reward = mean_reward
         self.last_improvement_step = self.n_calls
         self.stagnation_counter = 0
         best_path = self.results_dir / "models" / "best_model"
         self.model.save(str(best_path))
         self.custom_logger.info(f"💾 Nouveau meilleur modèle: {mean_reward:.2f}")
     else:
         self.stagnation_counter += 1
         
     # Détection de stagnation et redémarrage adaptatif
     if (self.n_calls - self.last_improvement_step) > self.stagnation_threshold:
         self._handle_stagnation()
     
     # Curriculum learning
     self._check_curriculum_advancement(mean_reward)
     
     self.custom_logger.info(
         f"   Reward: {mean_reward:.2f}, Contacts: {mean_contacts:.1f}, "
         f"Distance: {mean_distance:.3f}"
     )
 
 def _check_curriculum_advancement(self, eval_reward):
     """Vérifie et avance le curriculum si possible"""
     
     try:
         # Essayer d'avancer le curriculum dans l'environnement de training
         env = self.training_env
         
         # Déballage pour accéder à l'environnement de base
         while hasattr(env, 'env'):
             env = env.env
         
         if hasattr(env, 'advance_curriculum_level'):
             advanced = env.advance_curriculum_level(eval_reward)
             if advanced:
                 new_level = env.curriculum_level
                 self.curriculum_history.append({
                     'step': self.n_calls,
                     'level': new_level,
                     'reward': eval_reward
                 })
                 self.custom_logger.info(f"🎓 Curriculum avancé au niveau {new_level}")
                 
     except Exception as e:
         self.custom_logger.warning(f"Curriculum non disponible: {e}")
 
 def _handle_stagnation(self):
     """Gère la stagnation en augmentant l'exploration"""
     
     self.custom_logger.warning(f"⚠️ Stagnation détectée après {self.stagnation_threshold} steps")
     
     try:
         # Augmenter le bruit d'exploration
         if hasattr(self.model, 'action_noise') and self.model.action_noise is not None:
             current_sigma = self.model.action_noise.sigma
             new_sigma = np.minimum(current_sigma * 1.5, 0.5)  # Augmenter mais limiter
             self.model.action_noise.sigma = new_sigma
             self.custom_logger.info(f"🔄 Bruit d'exploration augmenté: {current_sigma[0]:.3f} → {new_sigma[0]:.3f}")
         
         # Réinitialiser le compteur
         self.last_improvement_step = self.n_calls
         self.stagnation_counter = 0
         
     except Exception as e:
         self.custom_logger.error(f"❌ Erreur gestion stagnation: {e}")
 
 def _create_video(self):
     """Création de vidéo comme le collègue"""
     
     if not self.eval_env or not imageio:
         return
     
     self.custom_logger.info(f"🎥 Création vidéo à {self.n_calls} steps...")
     
     try:
         obs, _ = self.eval_env.reset()
         frames = []
         
         for step in range(300):  # Comme le collègue
             action, _ = self.model.predict(obs, deterministic=True)
             obs, _, terminated, _, _ = self.eval_env.step(action)
             
             # Capturer frame
             frame = self.eval_env.render()
             if frame is not None:
                 frames.append(Image.fromarray(frame.astype(np.uint8)))
             
             if terminated:
                 break
         
         # Sauvegarder vidéo
         if frames:
             timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
             video_path = self.results_dir / "videos" / f"eval_{self.n_calls}_{timestamp}.mp4"
             imageio.mimsave(str(video_path), frames, fps=30)
             self.custom_logger.info(f"✅ Vidéo sauvegardée: {video_path.name}")
             
     except Exception as e:
         self.custom_logger.error(f"❌ Erreur création vidéo: {e}")
 
 def _save_model(self):
     """Sauvegarde périodique du modèle"""
     
     try:
         timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
         model_path = self.results_dir / "models" / f"model_{self.n_calls}_{timestamp}"
         self.model.save(str(model_path))
         self.custom_logger.info(f"💾 Modèle sauvegardé: {model_path.name}")
     except Exception as e:
         self.custom_logger.error(f"❌ Erreur sauvegarde: {e}")


# Dans optimized_training.py, modifiez la section de création de l'environnement :

def main():
 """
 🚀 TRAINING PRINCIPAL OPTIMISÉ
 """
 
 # Configuration inspirée du collègue mais équilibrée
 # CORRECTION 2: Paramètres training plus agressifs dans optimized_training.py

 config = {
     'total_timesteps': 100_000,    # Plus de temps pour converger
     'learning_rate': 3e-4,         # Learning rate standard et stable
     'batch_size': 256,             # Batch plus grand pour stabilité
     'buffer_size': 200_000,        # Buffer plus grand pour diversité
     'gamma': 0.99,                 # Horizon long pour planification
     'tau': 0.005,                  # Mise à jour douce des réseaux target
     'noise_std': 0.2,              # Exploration modérée mais efficace
     'results_dir': "retry_results"
}
 
 print("🚀 DÉMARRAGE DU TRAINING OPTIMISÉ")
 print("=" * 50)
 print("📋 Configuration:")
 for key, value in config.items():
     print(f"   {key}: {value}")
 print("=" * 50)
 
 # Créer environnement G1 EXCLUSIVEMENT
 print("🏗️ Création de l'environnement G1...")
 
 # FORCER g1_combined.xml - AUCUN FALLBACK
 model_path = "/home/oussema/Documents/project/results/g1_combined.xml"
 print(f"🤖 Utilisation EXCLUSIVE de: {model_path}")
 
 if not os.path.exists(model_path):
     print(f"❌ ERREUR CRITIQUE: {model_path} introuvable!")
     return
 
 # Créer l'environnement G1
 env = OptimizedGraspEnv(
     model_path=model_path,
     render_mode="rgb_array",
     max_episode_steps=500,
     curriculum_level=1,
     enable_smooth_movements=True
 )
 print("✅ Environnement G1 créé avec succès")
 
 # Wrapping avec Monitor
 env = Monitor(env)
 print("✅ Monitor appliqué")
 
 # Configuration du bruit comme le collègue
 print("🔧 Configuration du bruit d'action...")
 n_actions = env.action_space.shape[0]
 action_noise = NormalActionNoise(
     mean=np.zeros(n_actions), 
     sigma=config['noise_std'] * np.ones(n_actions)
 )

 # ← AJOUTEZ CES LIGNES POUR PRÉ-REMPLIR LE BUFFER
 print("🎲 Pré-remplissage du buffer avec actions aléatoires...")
 obs, _ = env.reset()
 for _ in range(1000):  # 1000 actions aléatoires
     random_action = env.action_space.sample()
     obs, _, terminated, _, _ = env.step(random_action)
     if terminated:
         obs, _ = env.reset()
 print("✅ Buffer pré-rempli")
 
 # Créer le modèle TD3 comme le collègue
 print("🧠 Création du modèle TD3...")
 model = TD3(
     "MlpPolicy",
     env,
     action_noise=action_noise,
     learning_rate=config['learning_rate'],
     batch_size=config['batch_size'],
     buffer_size=config['buffer_size'],
     gamma=config['gamma'],
     tau=config['tau'],
     verbose=1
 )
 
 print("✅ Modèle TD3 créé avec succès")
 print(f"   Device: {model.device}")
 print(f"   Action space: {env.action_space}")
 print(f"   Observation space: {env.observation_space}")
 
 # Callback optimisé
 print("🔄 Configuration des callbacks...")
 callback = OptimizedTrainingCallback(
     eval_freq=10000,   # Évaluation plus fréquente pour détecter stagnation
     video_freq=25000,  # Vidéos moins fréquentes
     save_freq=15000,   # Sauvegardes plus fréquentes
     n_eval_episodes=5, # Plus d'épisodes pour évaluation robuste
     results_dir=config['results_dir']
 )
 
 # Le reste du code reste identique...
 # TRAINING PRINCIPAL
 print("🎯 DÉBUT DU TRAINING OPTIMISÉ")
 print("=" * 50)
 
 start_time = time.time()
 
 try:
     # Test initial
     print("🧪 Test initial de l'environnement...")
     obs, _ = env.reset()
     print(f"   Observation shape: {obs.shape}")
     print(f"   Observation dtype: {obs.dtype}")
     
     # Training comme le collègue
     model.learn(
         total_timesteps=config['total_timesteps'],
         callback=callback,
         progress_bar=True
     )
     
     training_time = time.time() - start_time
     
     print("🏁 TRAINING TERMINÉ AVEC SUCCÈS!")
     print("=" * 50)
     print(f"⏱️  Temps total: {training_time/60:.1f} minutes")
     print(f"📊 Timesteps: {config['total_timesteps']:,}")
     
     # Sauvegarde finale
     final_model_path = Path(config['results_dir']) / "models" / "final_model"
     model.save(str(final_model_path))
     print(f"💾 Modèle final sauvegardé: {final_model_path}")
     
     # Évaluation finale approfondie
     print("🔍 ÉVALUATION FINALE...")
     final_evaluation(model, env, config['results_dir'])
     
 except KeyboardInterrupt:
     print("⏹️ Training interrompu par l'utilisateur")
 except Exception as e:
     print(f"❌ Erreur pendant le training: {e}")
     import traceback
     traceback.print_exc()
 finally:
     # Nettoyage
     try:
         env.close()
         if callback.eval_env:
             callback.eval_env.close()
     except:
         pass
     print("🧹 Nettoyage terminé")

def final_evaluation(model, env, results_dir: str):
 """
 Évaluation finale approfondie avec vidéo longue
 """
 
 print("📊 Évaluation finale sur 10 épisodes...")
 
 results_path = Path(results_dir)
 eval_rewards = []
 eval_contacts = []
 eval_distances = []
 
 # Évaluation détaillée
 for episode in range(10):
     obs, _ = env.reset()
     episode_reward = 0
     max_contacts = 0
     final_distance = float('inf')
     
     for step in range(500):
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, terminated, _, info = env.step(action)
         
         episode_reward += reward
         max_contacts = max(max_contacts, info.get('contact_count', 0))
         final_distance = info.get('distance', final_distance)
         
         if terminated:
             break
     
     eval_rewards.append(episode_reward)
     eval_contacts.append(max_contacts)
     eval_distances.append(final_distance)
     
     print(f"   Épisode {episode+1}: reward={episode_reward:.1f}, "
         f"contacts={max_contacts}, dist={final_distance:.3f}")
 
 # Statistiques finales
 mean_reward = np.mean(eval_rewards)
 mean_contacts = np.mean(eval_contacts)
 mean_distance = np.mean(eval_distances)
 success_rate = sum(1 for c in eval_contacts if c >= 2) / len(eval_contacts) * 100
 
 print("📈 RÉSULTATS FINAUX:")
 print(f"   Reward moyen: {mean_reward:.2f} ± {np.std(eval_rewards):.2f}")
 print(f"   Contacts moyens: {mean_contacts:.1f}")
 print(f"   Distance finale: {mean_distance:.3f} m")
 print(f"   Taux de succès: {success_rate:.1f}% (≥2 contacts)")
 
 # Vidéo finale longue comme le collègue
 if imageio:
     print("🎥 Création de la vidéo finale...")
     create_final_video(model, env, results_path)
 
 # Sauvegarder les statistiques
 stats = {
     'final_eval_reward': mean_reward,
     'final_eval_contacts': mean_contacts,
     'final_eval_distance': mean_distance,
     'success_rate': success_rate,
     'eval_rewards': eval_rewards,
     'eval_contacts': eval_contacts,
     'eval_distances': eval_distances
 }
 
 import json
 with open(results_path / "final_stats.json", 'w') as f:
     json.dump(stats, f, indent=2)
 
 print(f"💾 Statistiques sauvegardées: final_stats.json")


def create_final_video(model, env, results_dir: Path):
 """
 Crée une vidéo finale longue comme le collègue
 """
 
 try:
     obs, _ = env.reset()
     frames = []
     
     print("   Enregistrement de 1000 frames...")
     for step in range(1000):  # Comme le collègue
         action, _ = model.predict(obs, deterministic=True)
         obs, _, terminated, _, info = env.step(action)
         
         # Capturer frame
         frame = env.render()
         if frame is not None:
             frames.append(Image.fromarray(frame.astype(np.uint8)))
         
         # Afficher progrès
         if step % 200 == 0:
             dist = info.get('distance', 0)
             contacts = info.get('contact_count', 0)
             print(f"      Frame {step}: dist={dist:.3f}, contacts={contacts}")
         
         if terminated:
             break
     
     # Sauvegarder à 30 fps comme le collègue
     if frames:
         video_path = results_dir / "videos" / "final_evaluation.mp4"
         imageio.mimsave(str(video_path), frames, fps=30)
         print(f"✅ Vidéo finale sauvegardée: {video_path}")
     
 except Exception as e:
     print(f"❌ Erreur création vidéo finale: {e}")

if __name__ == "__main__":
 
 
 # Lancer le training principal
 main()
