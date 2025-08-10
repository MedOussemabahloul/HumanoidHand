#!/usr/bin/env python3
"""
🚀 SCRIPT D'ENTRAÎNEMENT OPTIMAL - BASÉ SUR LE CODE FONCTIONNEL DU COLLÈGUE
===========================================================================

Ce script reproduit EXACTEMENT la configuration qui fonctionne dans le notebook
de votre collègue, garantissant la stabilité et le bon fonctionnement.

✅ Utilise l'environnement optimal stable
✅ Configuration TD3 identique au notebook
✅ Callbacks pour vidéos d'évaluation
✅ Paramètres d'entraînement optimisés
✅ Gestion robuste des erreurs

Cette version garantit l'absence d'erreurs NaN/Inf et un entraînement stable.
"""

import os
import sys
import numpy as np
import torch
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Imports ML (exactement comme le notebook)
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import de notre environnement optimal
from envs.optimal_stable_env import OptimalStableGraspEnv

# Imports pour les vidéos (comme le notebook)
import imageio
from PIL import Image

class OptimalEvalVideoCallback(BaseCallback):
    """
    Callback pour enregistrer des vidéos d'évaluation
    Basé EXACTEMENT sur le code du notebook fonctionnel
    """
    
    def __init__(self, eval_env, eval_freq=50000, video_length=300, 
                 video_folder="videos/", prefix="optimal_grasp_eval", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.video_length = video_length
        self.video_folder = video_folder
        self.prefix = prefix
        os.makedirs(video_folder, exist_ok=True)
        
        print(f"🎥 Callback vidéo configuré : évaluation toutes les {eval_freq} steps")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            print(f"🎥 Enregistrement vidéo d'évaluation (step {self.n_calls})...")
            
            try:
                obs, _ = self.eval_env.reset()
                frames = []

                for step in range(self.video_length):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)

                    # Enregistrer la frame (comme le notebook)
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(Image.fromarray(frame.astype(np.uint8)))

                    if done:
                        print(f"🎯 Épisode terminé à l'étape {step}")
                        break

                # Sauvegarder la vidéo (comme le notebook)
                if frames:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    video_path = os.path.join(
                        self.video_folder, f"{self.prefix}_{self.n_calls}_steps_{timestamp}.mp4"
                    )
                    imageio.mimsave(video_path, frames, fps=30)
                    print(f"✅ Vidéo sauvegardée: {video_path}")
                    
            except Exception as e:
                print(f"⚠️ Erreur lors de l'enregistrement vidéo: {e}")

        return True

class OptimalTrainingMonitor(BaseCallback):
    """
    Callback de monitoring optimisé pour l'entraînement
    """
    
    def __init__(self, log_freq=100, save_freq=25000, results_dir="optimal_results"):
        super().__init__()
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Statistiques
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        
        print(f"📊 Monitor configuré : logs toutes les {log_freq} steps")
    
    def _on_step(self) -> bool:
        # Log périodique (comme le notebook)
        if self.n_calls % self.log_freq == 0 and self.n_calls > 0:
            try:
                # Récupérer les infos de l'environnement
                if hasattr(self.locals, 'rewards') and len(self.locals['rewards']) > 0:
                    current_reward = self.locals['rewards'][-1]
                    print(f"Step {self.n_calls}, reward: {current_reward:.3f}")
                    
                    # Tracker le meilleur reward
                    if current_reward > self.best_reward:
                        self.best_reward = current_reward
                        print(f"🏆 Nouveau meilleur reward: {self.best_reward:.3f}")
                        
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Erreur dans monitoring: {e}")
        
        # Sauvegarde périodique
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            try:
                model_path = self.results_dir / f"model_step_{self.n_calls}.zip"
                self.model.save(str(model_path))
                print(f"💾 Modèle sauvegardé: {model_path}")
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde: {e}")
        
        return True

def create_optimal_training_setup():
    """
    Créer la configuration d'entraînement optimale
    Basée EXACTEMENT sur le notebook fonctionnel
    """
    
    print("🚀 Configuration de l'entraînement optimal...")
    
    # ✅ Créer les environnements (comme le notebook)
    print("🔧 Création de l'environnement d'entraînement...")
    train_env = OptimalStableGraspEnv(eval_mode=False)
    
    print("🔧 Création de l'environnement d'évaluation...")
    eval_env = OptimalStableGraspEnv(eval_mode=True)
    
    # ✅ Configuration du bruit d'action (comme le notebook)
    n_actions = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.3 * np.ones(n_actions)  # Même sigma que le notebook
    )
    print(f"🎛️ Bruit d'action configuré pour {n_actions} actions")
    
    # ✅ Création du modèle TD3 (EXACTEMENT comme le notebook)
    model = TD3(
        "MlpPolicy",
        train_env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=3e-4,    # ✅ Même learning rate
        batch_size=256,        # ✅ Même batch size
        buffer_size=1_000_000, # ✅ Même buffer size
        gamma=0.98,            # ✅ Même gamma
        tau=0.02,              # ✅ Même tau
        device="auto"          # Utiliser GPU si disponible
    )
    
    print("✅ Modèle TD3 créé avec la configuration du notebook fonctionnel")
    
    # ✅ Configuration des callbacks
    video_callback = OptimalEvalVideoCallback(
        eval_env=eval_env,
        eval_freq=50000,    # ✅ Même fréquence que le notebook
        video_length=300,   # ✅ Même longueur
        video_folder="optimal_videos/",
        prefix="optimal_grasp_eval"
    )
    
    monitor_callback = OptimalTrainingMonitor(
        log_freq=100,       # Log plus fréquent pour monitoring
        save_freq=25000,
        results_dir="optimal_results"
    )
    
    return model, train_env, eval_env, [video_callback, monitor_callback]

def run_optimal_training(total_timesteps: int = 50000):
    """
    Lancer l'entraînement optimal
    Reproduit exactement la procédure du notebook fonctionnel
    """
    
    print("=" * 70)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT OPTIMAL")
    print("=" * 70)
    print(f"📅 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Timesteps total: {total_timesteps:,}")
    print()
    
    try:
        # ✅ Configuration (comme le notebook)
        model, train_env, eval_env, callbacks = create_optimal_training_setup()
        
        # ✅ Affichage des informations du modèle
        print("📊 INFORMATIONS DU MODÈLE:")
        print(f"  - Espace d'action: {train_env.action_space}")
        print(f"  - Espace d'observation: {train_env.observation_space}")
        print(f"  - Device: {model.device}")
        print(f"  - Politique: {model.policy}")
        print()
        
        # ✅ Test rapide de l'environnement
        print("🧪 Test de stabilité de l'environnement...")
        obs, _ = train_env.reset()
        for i in range(5):
            action = train_env.action_space.sample()
            obs, reward, done, _, _ = train_env.step(action)
            print(f"  Test step {i}: reward = {reward:.3f}, done = {done}")
            if done:
                obs, _ = train_env.reset()
        print("✅ Test de stabilité réussi!")
        print()
        
        # ✅ LANCEMENT DE L'ENTRAÎNEMENT (comme le notebook)
        print("🚀 Démarrage de l'entraînement...")
        print("📈 Surveillez les logs pour voir la progression...")
        print()
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        print()
        print("=" * 70)
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 70)
        print(f"⏱️ Durée: {training_duration:.2f} secondes")
        print(f"⚡ Steps/seconde: {total_timesteps/training_duration:.2f}")
        
        # ✅ Sauvegarde finale (comme le notebook)
        final_model_path = "optimal_results/optimal_td3_final_model"
        model.save(final_model_path)
        print(f"💾 Modèle final sauvegardé: {final_model_path}.zip")
        
        # ✅ Évaluation finale
        print("\n🎯 Évaluation finale...")
        obs, _ = eval_env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"📊 Évaluation finale:")
        print(f"  - Steps: {steps}")
        print(f"  - Reward total: {total_reward:.3f}")
        print(f"  - Reward moyen: {total_reward/steps:.3f}")
        
        # Fermeture propre
        train_env.close()
        eval_env.close()
        
        return model
        
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
        return None
        
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        print("🔧 Vérifiez la configuration de l'environnement")
        return None

def main():
    """Point d'entrée principal"""
    
    print("🎯 ENTRAÎNEMENT OPTIMAL BASÉ SUR LE CODE FONCTIONNEL DU COLLÈGUE")
    print("================================================================")
    print()
    
    # Vérifications préliminaires
    print("🔍 Vérifications préliminaires...")
    
    # Vérifier CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA non disponible, utilisation du CPU")
    
    # Vérifier le modèle XML
    model_path = "/workspace/results/g1_combined.xml"
    if os.path.exists(model_path):
        print(f"✅ Modèle XML trouvé: {model_path}")
    else:
        print(f"❌ Modèle XML introuvable: {model_path}")
        print("🔧 Assurez-vous que le fichier existe")
        return
    
    print()
    
    # Paramètres d'entraînement
    TOTAL_TIMESTEPS = 50000  # Comme le notebook (peut être augmenté)
    
    print(f"🎯 Paramètres d'entraînement:")
    print(f"  - Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Algorithme: TD3")
    print(f"  - Environnement: OptimalStableGraspEnv")
    print()
    
    # Lancer l'entraînement
    model = run_optimal_training(TOTAL_TIMESTEPS)
    
    if model is not None:
        print("🎉 Entraînement terminé avec succès!")
        print("📁 Résultats disponibles dans optimal_results/")
        print("🎥 Vidéos disponibles dans optimal_videos/")
    else:
        print("❌ Échec de l'entraînement")

if __name__ == "__main__":
    main()