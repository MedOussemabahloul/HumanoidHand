#!/usr/bin/env python3
"""
🎯 ENTRAÎNEMENT SAC POUR GRASPING G1
===================================

Script d'entraînement final utilisant:
- g1_combined.xml (cube fixe sur table)
- Agent SAC optimisé
- Génération automatique de vidéos
- Curriculum learning automatique
"""

import os
import sys
import numpy as np
import time
import json
import argparse
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.append('.')

def main():
    """Fonction principale d'entraînement"""
    
    print("🎯 ENTRAÎNEMENT SAC GRASPING G1")
    print("=" * 50)
    print("🤖 Robot G1 avec cube fixe sur table")
    print("🧠 Agent: Soft Actor-Critic (SAC)")
    print("🎬 Vidéos: Génération automatique")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='🎯 Entraînement SAC Grasping')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Nombre de timesteps (défaut: 100K)')
    parser.add_argument('--quick', action='store_true',
                       help='Test rapide 5K timesteps')
    parser.add_argument('--results-dir', type=str, default='sac_results',
                       help='Dossier de résultats')
    
    args = parser.parse_args()
    
    if args.quick:
        args.timesteps = 5000
        print("⚡ MODE RAPIDE - 5K timesteps")
    
    print(f"📊 Configuration:")
    print(f"   🎯 Timesteps: {args.timesteps:,}")
    print(f"   📁 Résultats: {args.results_dir}")
    print("=" * 50)
    
    try:
        # Imports
        from grasp_env import GraspEnv
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Créer les dossiers
        models_dir = os.path.join(args.results_dir, "models")
        videos_dir = os.path.join(args.results_dir, "videos")
        logs_dir = os.path.join(args.results_dir, "logs")
        
        for dir_path in [args.results_dir, models_dir, videos_dir, logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print("📁 Dossiers créés")
        
        # Callback de monitoring
        class ProgressCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_count = 0
                self.best_reward = -np.inf
                self.episode_rewards = []
                
            def _on_step(self):
                if len(self.locals.get('infos', [])) > 0:
                    info = self.locals['infos'][0]
                    if 'episode' in info:
                        self.episode_count += 1
                        reward = info['episode']['r']
                        self.episode_rewards.append(reward)
                        
                        if self.episode_count % 5 == 0:
                            print(f"📊 Épisode {self.episode_count:4d} | Récompense: {reward:7.2f}")
                        
                        if reward > self.best_reward:
                            self.best_reward = reward
                            model_path = os.path.join(models_dir, "best_model.zip")
                            self.model.save(model_path)
                            if self.episode_count % 10 == 0:
                                print(f"💾 Nouveau record: {reward:.2f}")
                
                return True
        
        # Créer l'environnement d'entraînement (sans vidéo pour performance)
        def make_env():
            env = GraspEnv(render_mode=None, record_video=False)
            return Monitor(env, os.path.join(logs_dir, "monitor.csv"))
        
        print("🏗️  Création de l'environnement...")
        env = DummyVecEnv([make_env])
        
        print("🧠 Création du modèle SAC...")
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            use_sde=True,
            verbose=1,
            device='cpu',
            tensorboard_log=logs_dir
        )
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"   🎯 Le robot va apprendre le grasping en {args.timesteps:,} steps")
        print(f"   📚 Phases: SEARCH → APPROACH → CONTACT → ALIGN → GRASP → LIFT → HOLD")
        print()
        
        start_time = time.time()
        
        # Callback de progression
        callback = ProgressCallback()
        
        # Entraînement
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        # Sauvegarder le modèle final
        final_model_path = os.path.join(models_dir, "final_model.zip")
        model.save(final_model_path)
        
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"⏱️  Durée: {training_time/60:.1f} minutes")
        print(f"📊 Épisodes: {callback.episode_count}")
        print(f"🏆 Meilleure récompense: {callback.best_reward:.2f}")
        
        # GÉNÉRATION AUTOMATIQUE DES VIDÉOS
        print(f"\n🎬 GÉNÉRATION AUTOMATIQUE DES VIDÉOS...")
        
        # Charger le meilleur modèle
        best_model_path = os.path.join(models_dir, "best_model.zip")
        if os.path.exists(best_model_path):
            demo_model = SAC.load(best_model_path)
            print(f"✅ Meilleur modèle chargé")
        else:
            demo_model = model
            print(f"✅ Modèle final utilisé")
        
        # Créer environnement avec vidéo
        demo_env = GraspEnv(render_mode="rgb_array", record_video=True, video_dir=videos_dir)
        
        # Générer 3 vidéos de démonstration
        video_rewards = []
        for episode in range(3):
            print(f"🎬 Enregistrement vidéo {episode + 1}/3...")
            
            obs, _ = demo_env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(500):
                action, _ = demo_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = demo_env.step(action)
                total_reward += reward
                steps += 1
                
                if step % 100 == 0:
                    phase = info.get('phase', 'UNKNOWN')
                    print(f"   Step {step:3d} | Phase: {phase:8s} | Reward: {reward:6.2f}")
                
                if terminated or truncated:
                    break
            
            video_rewards.append(total_reward)
            print(f"   ✅ Vidéo terminée - Reward: {total_reward:.2f} ({steps} steps)")
            
            # Sauvegarder cette vidéo
            demo_env.save_video(f"demo_episode_{episode + 1:02d}_reward_{total_reward:.0f}.mp4")
        
        demo_env.close()
        print(f"✅ 3 vidéos générées automatiquement!")
        
        # Créer un rapport final
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_time_minutes': training_time / 60,
            'total_timesteps': args.timesteps,
            'total_episodes': callback.episode_count,
            'best_reward': float(callback.best_reward),
            'mean_reward': float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
            'video_rewards': video_rewards,
            'files': {
                'best_model': os.path.join(models_dir, "best_model.zip"),
                'final_model': os.path.join(models_dir, "final_model.zip"),
                'videos_dir': videos_dir,
                'logs_dir': logs_dir
            }
        }
        
        report_path = os.path.join(args.results_dir, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Affichage final
        print(f"\n📋 RAPPORT FINAL:")
        print(f"   ⏱️  Temps d'entraînement: {training_time/60:.1f} minutes")
        print(f"   📊 Total épisodes: {callback.episode_count}")
        print(f"   🏆 Meilleure récompense: {callback.best_reward:.2f}")
        print(f"   📈 Récompense moyenne: {np.mean(callback.episode_rewards):.2f}")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        print(f"   🧠 Meilleur modèle: models/best_model.zip")
        print(f"   🧠 Modèle final: models/final_model.zip")
        print(f"   🎬 Vidéos: videos/demo_episode_*.mp4")
        print(f"   📊 Logs: logs/monitor.csv")
        print(f"   📋 Rapport: training_report.json")
        
        print(f"\n🎬 VIDÉOS GÉNÉRÉES:")
        for i, reward in enumerate(video_rewards):
            video_file = f"demo_episode_{i + 1:02d}_reward_{reward:.0f}.mp4"
            print(f"   📹 {video_file}")
        
        print(f"\n🎯 UTILISATION DU MODÈLE:")
        print(f"```python")
        print(f"from stable_baselines3 import SAC")
        print(f"from grasp_env import GraspEnv")
        print(f"")
        print(f"model = SAC.load('{models_dir}/best_model.zip')")
        print(f"env = GraspEnv(render_mode='rgb_array', record_video=True)")
        print(f"obs, _ = env.reset()")
        print(f"for _ in range(500):")
        print(f"    action, _ = model.predict(obs, deterministic=True)")
        print(f"    obs, reward, done, truncated, info = env.step(action)")
        print(f"    if done or truncated: break")
        print(f"env.save_video('test.mp4')")
        print(f"```")
        
        print(f"\n🎊 SUCCÈS COMPLET!")
        print(f"🤖 Robot G1 entraîné avec cube fixe sur table")
        print(f"🎬 Vidéos générées et téléchargées automatiquement")
        print(f"📁 Tous les fichiers dans: {args.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)