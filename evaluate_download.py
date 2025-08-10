#!/usr/bin/env python3
"""
🎬 SCRIPT D'ÉVALUATION ET TÉLÉCHARGEMENT VIDÉO
==============================================

Script autonome pour évaluer un modèle entraîné et générer une vidéo,
inspiré de la dernière cellule du collègue qui génère et télécharge les vidéos.
"""

import os
import sys
import numpy as np
import imageio
from PIL import Image
from pathlib import Path
import argparse

# Imports ML
from stable_baselines3 import TD3

# Import de notre environnement
from envs.simple_robust_grasp_env import SimpleRobustGraspEnv

def load_model(model_path):
 """Charger un modèle TD3 sauvegardé"""
 try:
     model = TD3.load(model_path)
     print(f"✅ Modèle chargé depuis: {model_path}")
     return model
 except Exception as e:
     print(f"❌ Erreur chargement modèle: {e}")
     return None

def evaluate_model(model, env, n_episodes=5, verbose=True):
 """
 Évaluer un modèle sur plusieurs épisodes
 """
 print(f"📊 Évaluation sur {n_episodes} épisodes...")
 
 episode_rewards = []
 episode_lengths = []
 successful_grasps = 0
 
 for episode in range(n_episodes):
     obs, info = env.reset()
     episode_reward = 0
     episode_length = 0
     episode_success = False
     
     for step in range(500):  # Max steps par épisode
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, terminated, truncated, info = env.step(action)
         
         episode_reward += reward
         episode_length += 1
         
         # Vérifier le succès
         if info.get('successful_grasp', False):
             episode_success = True
         
         if terminated or truncated:
             break
     
     episode_rewards.append(episode_reward)
     episode_lengths.append(episode_length)
     
     if episode_success:
         successful_grasps += 1
     
     if verbose:
         print(f"  Épisode {episode+1}: Reward={episode_reward:.2f}, "
               f"Length={episode_length}, Success={episode_success}")
 
 # Statistiques finales
 mean_reward = np.mean(episode_rewards)
 std_reward = np.std(episode_rewards)
 mean_length = np.mean(episode_lengths)
 success_rate = successful_grasps / n_episodes * 100
 
 print(f"\n📈 RÉSULTATS D'ÉVALUATION:")
 print(f"  Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")
 print(f"  Longueur moyenne: {mean_length:.1f} steps")
 print(f"  Taux de succès: {success_rate:.1f}% ({successful_grasps}/{n_episodes})")
 print(f"  Meilleur épisode: {max(episode_rewards):.2f}")
 
 return {
     'mean_reward': mean_reward,
     'std_reward': std_reward,
     'mean_length': mean_length,
     'success_rate': success_rate,
     'episode_rewards': episode_rewards,
     'episode_lengths': episode_lengths
 }

def create_evaluation_video(model, env, video_path="evaluation_video.mp4", 
                       n_steps=1000, fps=30):
 """
 Créer une vidéo d'évaluation longue
 Basé sur la dernière cellule du collègue
 """
 print(f"🎬 Création vidéo d'évaluation: {video_path}")
 print(f"📏 Durée: {n_steps} steps à {fps} fps = {n_steps/fps:.1f} secondes")
 
 frames = []
 obs, info = env.reset()
 
 total_reward = 0
 contacts_detected = 0
 successful_moments = 0
 
 for t in range(n_steps):
     # Prédiction déterministe comme le collègue
     action, _ = model.predict(obs, deterministic=True)
     obs, reward, terminated, truncated, info = env.step(action)
     
     # Statistiques
     total_reward += reward
     if info.get('total_contacts', 0) > 0:
         contacts_detected += 1
     if info.get('successful_grasp', False):
         successful_moments += 1
     
     # Capturer la frame
     frame = env.render()
     if frame is not None:
         frames.append(Image.fromarray(frame.astype(np.uint8)))
     
     # Progress indication
     if t % 100 == 0:
         print(f"  Progress: {t}/{n_steps} ({t/n_steps*100:.1f}%) - "
               f"Reward cumulé: {total_reward:.1f}")
     
     # Reset si nécessaire
     if terminated or truncated:
         obs, info = env.reset()
 
 # Sauvegarder la vidéo à 30 fps comme le collègue
 if frames:
     print(f"💾 Sauvegarde de {len(frames)} frames...")
     imageio.mimsave(video_path, frames, fps=fps)
     
     print(f"✅ Vidéo sauvegardée: {video_path}")
     print(f"📊 Reward total: {total_reward:.2f}")
     print(f"🤝 Moments avec contact: {contacts_detected}/{n_steps} ({contacts_detected/n_steps*100:.1f}%)")
     print(f"🏆 Moments de succès: {successful_moments}/{n_steps} ({successful_moments/n_steps*100:.1f}%)")
     
     # Informations sur le fichier
     file_size = os.path.getsize(video_path) / (1024*1024)  # MB
     print(f"📁 Taille du fichier: {file_size:.1f} MB")
     
     return True
 else:
     print("❌ Aucune frame capturée")
     return False

def download_video(video_path):
 """
 Simuler le téléchargement de vidéo (comme files.download() du collègue)
 En environnement local, on affiche juste le chemin
 """
 if os.path.exists(video_path):
     print(f"📥 Vidéo prête au téléchargement: {os.path.abspath(video_path)}")
     print(f"🔗 Chemin complet: {video_path}")
     
     # Tentative d'ouverture automatique (optionnel)
     try:
         import webbrowser
         webbrowser.open(f"file://{os.path.abspath(video_path)}")
     except:
         pass
     
     return True
 else:
     print(f"❌ Fichier vidéo non trouvé: {video_path}")
     return False

def main():
 """Fonction principale avec arguments en ligne de commande"""
 parser = argparse.ArgumentParser(description="Évaluation et génération de vidéo")
 parser.add_argument("--model", "-m", 
                    default="simple_td3_results/final_model.zip",
                    help="Chemin vers le modèle TD3 (.zip)")
 parser.add_argument("--video", "-v", 
                    default="evaluation_longue.mp4",
                    help="Nom du fichier vidéo de sortie")
 parser.add_argument("--steps", "-s", type=int, default=1000,
                    help="Nombre de steps pour la vidéo")
 parser.add_argument("--fps", type=int, default=30,
                    help="FPS de la vidéo")
 parser.add_argument("--eval-episodes", "-e", type=int, default=5,
                    help="Nombre d'épisodes pour l'évaluation")
 parser.add_argument("--no-eval", action="store_true",
                    help="Ignorer l'évaluation, juste créer la vidéo")
 
 args = parser.parse_args()
 
 print("🎬 ÉVALUATION ET GÉNÉRATION VIDÉO")
 print("=" * 50)
 print(f"📁 Modèle: {args.model}")
 print(f"🎥 Vidéo: {args.video}")
 print(f"📏 Steps: {args.steps}")
 print(f"🎞️ FPS: {args.fps}")
 
 # Vérifier que le modèle existe
 if not os.path.exists(args.model):
     print(f"❌ Modèle non trouvé: {args.model}")
     print("💡 Suggestions:")
     print("  - Lancez d'abord l'entraînement: python simple_training_td3.py")
     print("  - Ou spécifiez un autre modèle avec --model")
     return
 
 # Charger le modèle
 model = load_model(args.model)
 if model is None:
     return
 
 # Créer l'environnement d'évaluation
 print("🏗️ Création de l'environnement d'évaluation...")
 env = SimpleRobustGraspEnv(eval_mode=True)
 
 # Évaluation (optionnelle)
 if not args.no_eval:
     results = evaluate_model(model, env, args.eval_episodes)
 
 # Créer la vidéo longue
 success = create_evaluation_video(
     model, env, args.video, args.steps, args.fps
 )
 
 if success:
     # "Télécharger" la vidéo
     download_video(args.video)
     
     print(f"\n🎉 ÉVALUATION TERMINÉE!")
     print(f"🎥 Vidéo disponible: {args.video}")
     if not args.no_eval:
         print(f"📊 Performance: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
         print(f"🏆 Taux de succès: {results['success_rate']:.1f}%")
 
 # Fermeture propre
 env.close()
 print("✅ Évaluation terminée")

# Version rapide pour reproduire exactement le code du collègue
def quick_evaluation_like_colleague():
 """
 Fonction qui reproduit exactement la dernière cellule du collègue
 """
 print("🔄 ÉVALUATION RAPIDE (style collègue)")
 
 # Charger le modèle (adapter le chemin selon votre cas)
 model_path = "simple_td3_results/final_model.zip"
 if os.path.exists(model_path):
     model = TD3.load(model_path)
 else:
     print(f"❌ Modèle non trouvé: {model_path}")
     return
 
 # Créer l'environnement
 env = SimpleRobustGraspEnv(eval_mode=True)
 
 # 10. Evaluate and Record Video (longer) - comme le collègue
 frames = []
 obs, _ = env.reset()
 
 for t in range(1000):  # Augmenter le nombre de steps/frames
     action, _ = model.predict(obs, deterministic=True)
     obs, _, _, _, _ = env.step(action)
     frame = env.render()
     if frame is not None:
         frames.append(Image.fromarray(frame.astype(np.uint8)))
 
 # Sauvegarder à 30 fps
 video_filename = "custom_robot_eval.mp4"
 imageio.mimsave(video_filename, frames, fps=30)
 
 # Télécharger la vidéo (simulation de files.download())
 print(f"✅ Vidéo créée: {video_filename}")
 download_video(video_filename)
 
 env.close()

if __name__ == "__main__":
 # Offrir les deux options
 if len(sys.argv) == 1:
     print("🤔 Mode rapide (comme le collègue) ou mode complet?")
     print("1. Mode rapide: python evaluate_and_download.py --quick")
     print("2. Mode complet: python evaluate_and_download.py --help")
     print()
     print("🚀 Lancement en mode rapide par défaut...")
     quick_evaluation_like_colleague()
 elif "--quick" in sys.argv:
     quick_evaluation_like_colleague()
 else:
     main()
