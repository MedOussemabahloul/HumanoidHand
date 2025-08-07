#!/usr/bin/env python3
"""
🎯 ENTRAÎNEUR SAC OPTIMISÉ POUR GRASPING
=======================================

Version optimisée qui résout les problèmes de:
- Vitesses excessives constantes
- Récompenses négatives persistantes
- Mauvaise convergence
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
sys.path.append('/workspace/envs')

try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
    print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

class OptimizedProgressCallback(BaseCallback):
    """Callback optimisé pour suivre les progrès"""
    
    def __init__(self, check_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_count = 0
        self.best_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Enregistrer les récompenses d'épisode
        if self.locals.get('dones', [False])[0]:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_reward = self.locals['infos'][0]['episode']['r']
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1
                
                # Suivre le meilleur score
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    print(f"🎉 Nouveau record! Épisode {self.episode_count}: {episode_reward:.2f}")
                
                # Afficher progrès périodiquement
                if self.episode_count % 20 == 0:
                    recent_rewards = self.episode_rewards[-20:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"📊 Épisode {self.episode_count}: Moyenne (20 derniers): {avg_reward:.2f}, Meilleur: {self.best_reward:.2f}")
                    
                    # Vérifier si on progresse
                    if len(self.episode_rewards) >= 40:
                        old_avg = np.mean(self.episode_rewards[-40:-20])
                        new_avg = np.mean(self.episode_rewards[-20:])
                        if new_avg > old_avg:
                            print(f"📈 Progrès détecté: {old_avg:.2f} → {new_avg:.2f}")
                        else:
                            print(f"📉 Stagnation: {old_avg:.2f} → {new_avg:.2f}")
        
        return True

class OptimizedCurriculumTrainer:
    """Entraîneur optimisé avec paramètres ajustés"""
    
    def __init__(self, total_timesteps: int = 100000):
        self.total_timesteps = total_timesteps
        
        # Configuration des dossiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"/workspace/optimized_sac_results_{timestamp}"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        
        # Créer les dossiers
        for directory in [self.results_dir, self.models_dir, self.logs_dir, self.videos_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"🎯 OptimizedCurriculumTrainer initialisé")
        print(f"📁 Résultats: {self.results_dir}")
        
    def create_optimized_environment(self):
        """Créer l'environnement avec paramètres optimisés"""
        try:
            print("🏗️  Création de l'environnement optimisé...")
            self.env = CurriculumGraspEnv()
            
            # Forcer le niveau 1 pour commencer par la stabilisation
            self.env.current_level = 1
            self.env._update_curriculum_config()
            
            print("✅ Environnement optimisé créé")
            print(f"   📚 Niveau: {self.env.current_level} - {self.env.curriculum_levels[1]['name']}")
            return True
        except Exception as e:
            print(f"❌ Erreur création environnement: {e}")
            return False
    
    def create_optimized_model(self):
        """Créer le modèle SAC avec paramètres optimisés pour stabilisation"""
        try:
            print("🧠 Création du modèle SAC optimisé...")
            
            # Paramètres optimisés pour la stabilisation
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=0.0001,      # Plus lent pour stabilité
                buffer_size=50000,         # Buffer plus petit au début
                batch_size=128,            # Batch size plus petit
                tau=0.01,                  # Soft update plus agressif
                gamma=0.98,                # Discount factor plus faible
                train_freq=4,              # Entraîner moins souvent
                gradient_steps=2,          # Plus de gradient steps
                ent_coef=0.2,              # Exploration modérée
                target_update_interval=2,  # Update target plus souvent
                policy_kwargs=dict(
                    net_arch=[128, 128],   # Réseaux plus petits
                    activation_fn=lambda: __import__('torch.nn', fromlist=['Tanh']).Tanh()
                ),
                verbose=1,
                device="auto",
                tensorboard_log=self.logs_dir
            )
            
            # Configuration du logger
            logger = configure(self.logs_dir, ["stdout", "csv", "tensorboard"])
            self.model.set_logger(logger)
            
            print("✅ Modèle SAC optimisé créé")
            print(f"  - Learning rate: {self.model.learning_rate}")
            print(f"  - Buffer size: {self.model.buffer_size}")
            print(f"  - Batch size: {self.model.batch_size}")
            print(f"  - Tau: {self.model.tau}")
            print(f"  - Gamma: {self.model.gamma}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur création modèle: {e}")
            return False
    
    def train_with_optimization(self):
        """Entraînement optimisé avec monitoring"""
        try:
            print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT OPTIMISÉ")
            print("=" * 50)
            
            # Callback pour monitoring
            callback = OptimizedProgressCallback(check_freq=1000)
            
            # Entraînement par étapes pour permettre l'ajustement
            steps_per_phase = self.total_timesteps // 4
            
            for phase in range(4):
                print(f"\n📚 PHASE {phase + 1}/4 - {steps_per_phase} timesteps")
                print("-" * 30)
                
                start_time = time.time()
                
                self.model.learn(
                    total_timesteps=steps_per_phase,
                    callback=callback,
                    log_interval=5,
                    reset_num_timesteps=False
                )
                
                phase_time = time.time() - start_time
                print(f"⏱️  Phase {phase + 1} terminée en {phase_time:.1f}s")
                
                # Sauvegarder le modèle intermédiaire
                model_path = os.path.join(self.models_dir, f"optimized_sac_phase_{phase + 1}.zip")
                self.model.save(model_path)
                print(f"💾 Modèle phase {phase + 1} sauvé")
                
                # Vérifier les progrès
                if len(callback.episode_rewards) >= 20:
                    recent_avg = np.mean(callback.episode_rewards[-20:])
                    print(f"📊 Récompense moyenne récente: {recent_avg:.2f}")
                    
                    # Si on progresse bien, on peut ajuster les paramètres
                    if recent_avg > -50 and phase == 1:
                        print("🎯 Progrès détecté! Ajustement des paramètres...")
                        self.model.learning_rate = 0.0003  # Augmenter légèrement
                        self.model.ent_coef = 0.1          # Réduire exploration
            
            # Sauvegarder le modèle final
            final_model_path = os.path.join(self.models_dir, "optimized_sac_final.zip")
            self.model.save(final_model_path)
            print(f"💾 Modèle final sauvé: {final_model_path}")
            
            print(f"✅ Entraînement optimisé terminé!")
            print(f"📊 Total d'épisodes: {callback.episode_count}")
            print(f"🏆 Meilleur score: {callback.best_reward:.2f}")
            
            return True, callback
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_trained_model(self, num_episodes: int = 5):
        """Tester le modèle entraîné"""
        try:
            print("\n🎮 TEST DU MODÈLE ENTRAÎNÉ")
            print("=" * 40)
            
            test_rewards = []
            
            for episode in range(num_episodes):
                obs, info = self.env.reset()
                episode_reward = 0
                step_count = 0
                
                print(f"🎮 Épisode de test {episode + 1}/{num_episodes}")
                
                for step in range(500):  # Limite de steps
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                
                test_rewards.append(episode_reward)
                print(f"   Récompense: {episode_reward:.2f} (steps: {step_count})")
            
            avg_test_reward = np.mean(test_rewards)
            print(f"\n📊 RÉSULTATS DU TEST:")
            print(f"   Récompense moyenne: {avg_test_reward:.2f}")
            print(f"   Récompense min: {min(test_rewards):.2f}")
            print(f"   Récompense max: {max(test_rewards):.2f}")
            
            return avg_test_reward > -30  # Critère de succès ajusté
            
        except Exception as e:
            print(f"❌ Erreur test modèle: {e}")
            return False
    
    def generate_demo_video(self, num_episodes: int = 2):
        """Générer vidéo de démonstration"""
        try:
            print("\n🎬 GÉNÉRATION VIDÉO DE DÉMONSTRATION")
            print("=" * 45)
            
            video_env = CurriculumGraspEnv(render_mode='rgb_array')
            video_env.current_level = self.env.current_level
            video_env._update_curriculum_config()
            
            video_path = os.path.join(self.videos_dir, "optimized_demo.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20, (640, 480))
            
            total_frames = 0
            
            for episode in range(num_episodes):
                print(f"🎬 Enregistrement épisode {episode + 1}/{num_episodes}")
                
                obs, info = video_env.reset()
                
                for step in range(300):  # Épisodes plus courts pour vidéo
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = video_env.step(action)
                    
                    # Capturer frame
                    try:
                        frame = video_env.render()
                        if frame is not None:
                            if frame.shape[:2] != (480, 640):
                                frame = cv2.resize(frame, (640, 480))
                            if len(frame.shape) == 3:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame)
                            total_frames += 1
                    except:
                        pass
                    
                    if terminated or truncated:
                        break
            
            video_writer.release()
            video_env.close()
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"✅ Vidéo générée: {video_path}")
                print(f"🎬 Frames capturées: {total_frames}")
                return True
            else:
                print("⚠️  Vidéo non générée")
                return False
                
        except Exception as e:
            print(f"❌ Erreur génération vidéo: {e}")
            return False
    
    def cleanup(self):
        """Nettoyage"""
        if hasattr(self, 'env'):
            self.env.close()

def main():
    """Fonction principale optimisée"""
    print("🎯 LANCEMENT DE L'ENTRAÎNEMENT SAC OPTIMISÉ")
    print("=" * 60)
    
    # Configuration optimisée
    total_timesteps = 100000  # Entraînement plus long pour convergence
    
    trainer = OptimizedCurriculumTrainer(total_timesteps=total_timesteps)
    
    try:
        # 1. Créer l'environnement optimisé
        if not trainer.create_optimized_environment():
            print("❌ Échec création environnement")
            return
        
        # 2. Créer le modèle optimisé
        if not trainer.create_optimized_model():
            print("❌ Échec création modèle")
            return
        
        # 3. Entraîner avec optimisations
        success, callback = trainer.train_with_optimization()
        if not success:
            print("❌ Échec entraînement")
            return
        
        # 4. Tester le modèle
        if trainer.test_trained_model():
            print("✅ Modèle teste avec succès!")
            
            # 5. Générer vidéo si le test réussit
            trainer.generate_demo_video()
        else:
            print("⚠️  Modèle nécessite plus d'entraînement")
        
        print(f"\n🎉 ENTRAÎNEMENT OPTIMISÉ TERMINÉ!")
        print(f"📁 Résultats: {trainer.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()