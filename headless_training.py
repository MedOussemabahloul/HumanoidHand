#!/usr/bin/env python3
"""
🚀 TRAINING HEADLESS OPTIMISÉ
=============================

Script de training sans rendu pour éviter les problèmes EGL/OpenGL
Inspiré du collègue avec curriculum learning
"""

import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import logging

# Configuration des warnings
warnings.filterwarnings("ignore")

# Imports ML
try:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    print("✅ Stable-Baselines3 importé")
except ImportError as e:
    print(f"❌ Erreur import ML: {e}")
    sys.exit(1)

# Import de notre environnement headless
try:
    from envs.headless_optimized_grasp_env import HeadlessOptimizedGraspEnv
    print("✅ Environnement headless importé")
except ImportError:
    print("❌ ERREUR: Impossible d'importer l'environnement headless")
    sys.exit(1)


class HeadlessTrainingCallback(BaseCallback):
    """
    Callback pour training headless
    """
    
    def __init__(self, 
                 eval_freq: int = 25000,
                 save_freq: int = 25000,
                 n_eval_episodes: int = 5,
                 results_dir: str = "headless_results",
                 verbose: int = 1):
        
        super().__init__(verbose)
        
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.results_dir = Path(results_dir)
        
        # Créer les dossiers
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        
        # Métriques
        self.eval_rewards = []
        self.best_eval_reward = -np.inf
        self.curriculum_history = []
        
        # Environnement d'évaluation
        self.eval_env = None
        
        # Logger
        self.setup_logging()
    
    def setup_logging(self):
        """Configure le logging"""
        self.custom_logger = logging.getLogger("HeadlessTraining")
        self.custom_logger.setLevel(logging.INFO)
        
        if not self.custom_logger.handlers:
            # Handler fichier
            handler = logging.FileHandler(self.results_dir / "logs" / "training.log")
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
            self.eval_env = HeadlessOptimizedGraspEnv()
            self.custom_logger.info("✅ Environnement d'évaluation headless créé")
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur création env d'évaluation: {e}")
    
    def _on_step(self) -> bool:
        """Appelé à chaque step"""
        
        # Logging périodique
        if self.n_calls % 1000 == 0:
            try:
                infos = self.locals.get('infos', [{}])
                if infos and len(infos) > 0:
                    info = infos[0]
                    
                    # Log métriques de base
                    dist = info.get('distance', 0)
                    contacts = info.get('contact_count', 0)
                    curriculum = info.get('curriculum_level', 1)
                    
                    self.custom_logger.info(
                        f"Step {self.n_calls}: dist={dist:.3f}, "
                        f"contacts={contacts}, curriculum={curriculum}"
                    )
            except Exception as e:
                self.custom_logger.warning(f"Erreur logging: {e}")
        
        # Évaluation périodique
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            self._run_evaluation()
        
        # Sauvegarde périodique
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_model()
        
        return True
    
    def _run_evaluation(self):
        """Évaluation sans vidéo"""
        
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
        
        # Sauvegarder le meilleur modèle
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            best_path = self.results_dir / "models" / "best_model"
            self.model.save(str(best_path))
            self.custom_logger.info(f"💾 Nouveau meilleur modèle: {mean_reward:.2f}")
        
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
    
    def _save_model(self):
        """Sauvegarde périodique du modèle"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = self.results_dir / "models" / f"model_{self.n_calls}_{timestamp}"
            self.model.save(str(model_path))
            self.custom_logger.info(f"💾 Modèle sauvegardé: {model_path.name}")
        except Exception as e:
            self.custom_logger.error(f"❌ Erreur sauvegarde: {e}")


def main():
    """
    🚀 TRAINING PRINCIPAL HEADLESS
    """
    
    # Configuration inspirée du collègue
    config = {
        'total_timesteps': 100_000,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 1_000_000,
        'gamma': 0.98,
        'tau': 0.02,
        'noise_std': 0.3,
        'results_dir': "headless_results"
    }
    
    print("🚀 DÉMARRAGE DU TRAINING HEADLESS")
    print("=" * 50)
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 50)
    
    # Créer environnement principal
    print("🏗️ Création de l'environnement headless...")
    try:
        env = HeadlessOptimizedGraspEnv()
        env = Monitor(env)
        print("✅ Environnement headless créé avec succès")
    except Exception as e:
        print(f"❌ Erreur création environnement: {e}")
        return
    
    # Configuration du bruit comme le collègue
    print("🔧 Configuration du bruit d'action...")
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=config['noise_std'] * np.ones(n_actions)
    )
    
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
    
    # Callback headless
    print("🔄 Configuration des callbacks...")
    callback = HeadlessTrainingCallback(
        eval_freq=25000,
        save_freq=25000,
        n_eval_episodes=3,
        results_dir=config['results_dir']
    )
    
    # TRAINING PRINCIPAL
    print("🎯 DÉBUT DU TRAINING HEADLESS")
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
        
        # Évaluation finale
        print("🔍 ÉVALUATION FINALE...")
        final_evaluation_headless(model, env, config['results_dir'])
        
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


def final_evaluation_headless(model, env, results_dir: str):
    """
    Évaluation finale sans vidéo
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


if __name__ == "__main__":
    main()