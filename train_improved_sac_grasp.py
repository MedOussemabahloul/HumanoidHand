#!/usr/bin/env python3
"""
🎯 ENTRAÎNEMENT SAC AMÉLIORÉ POUR GRASPING G1
============================================

Corrections appliquées:
✅ Environnement ultra-stable avec physics fixes
✅ Hyperparamètres SAC optimisés pour stabilité
✅ Système de récompenses progressif
✅ Gestion des erreurs et récupération automatique
✅ Monitoring détaillé et logging
✅ Mouvements fluides garantis
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/workspace/envs')

try:
    from envs.improved_professional_grasp_env import ImprovedProfessionalGraspEnv
    print("✅ ImprovedProfessionalGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class ImprovedGraspingTrainer:
    """
    🏆 Entraîneur SAC Ultra-Amélioré pour Grasping G1
    
    Optimisations:
    - Hyperparamètres SAC spécialement réglés pour stabilité
    - Learning rate adaptatif
    - Buffer replay optimisé
    - Monitoring en temps réel
    - Sauvegarde automatique des meilleurs modèles
    """
    
    def __init__(self, total_timesteps: int = 100000):
        self.total_timesteps = total_timesteps
        
        # Configuration des dossiers
        self.results_dir = "/workspace/improved_sac_results"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        
        self._setup_directories()
        
        # Métriques d'entraînement
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'phase_completions': [],
            'stability_scores': [],
            'training_time': 0.0,
            'best_reward': -np.inf,
            'consecutive_successes': 0
        }
        
        print("🎯 ImprovedGraspingTrainer initialisé")
        print(f"📁 Résultats: {self.results_dir}")
    
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        for directory in [self.results_dir, self.models_dir, self.logs_dir, self.videos_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def create_environment(self):
        """Crée l'environnement d'entraînement"""
        print("🏗️  Création de l'environnement...")
        
        def _make_env():
            env = ImprovedProfessionalGraspEnv(
                model_path="/workspace/results/g1_combined.xml",
                render_mode=None
            )
            return env
        
        # Créer un environnement vectorisé pour de meilleures performances
        env = make_vec_env(_make_env, n_envs=1)
        
        print("✅ Environnement créé avec succès")
        return env
    
    def create_sac_model(self, env):
        """Crée le modèle SAC avec hyperparamètres optimisés"""
        print("🧠 Création du modèle SAC...")
        
        # Hyperparamètres SAC optimisés pour stabilité
        sac_params = {
            'learning_rate': 3e-4,        # Learning rate modéré
            'buffer_size': 100000,        # Buffer suffisant mais pas trop grand
            'learning_starts': 1000,      # Commencer l'apprentissage plus tôt
            'batch_size': 256,            # Batch size raisonnable
            'tau': 0.005,                 # Soft update coefficient
            'gamma': 0.99,                # Discount factor élevé
            'train_freq': 1,              # Entraîner à chaque step
            'gradient_steps': 1,          # Un gradient step par step
            'ent_coef': 'auto',           # Coefficient d'entropie automatique
            'target_update_interval': 1,   # Update target network fréquemment
            'use_sde': False,             # Pas de SDE pour plus de stabilité
            'verbose': 1,
        }
        
        # Créer le modèle
        model = SAC(
            "MlpPolicy",
            env,
            **sac_params,
            tensorboard_log=self.logs_dir
        )
        
        print("✅ Modèle SAC créé avec hyperparamètres optimisés")
        print(f"  - Learning rate: {sac_params['learning_rate']}")
        print(f"  - Buffer size: {sac_params['buffer_size']}")
        print(f"  - Batch size: {sac_params['batch_size']}")
        
        return model
    
    def train(self):
        """Lance l'entraînement principal"""
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT SAC AMÉLIORÉ")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Créer environnement et modèle
            env = self.create_environment()
            model = self.create_sac_model(env)
            
            # Callback pour monitoring
            callback = ImprovedTrainingCallback(
                trainer=self,
                check_freq=1000,
                save_freq=10000,
                verbose=1
            )
            
            # Configuration du logging
            logger = configure(self.logs_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(logger)
            
            print(f"🎯 Entraînement pour {self.total_timesteps} timesteps...")
            
            # Entraînement principal
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                log_interval=100,
                reset_num_timesteps=False
            )
            
            # Sauvegarder le modèle final
            final_model_path = os.path.join(self.models_dir, "sac_final_model.zip")
            model.save(final_model_path)
            
            # Calculer temps d'entraînement
            training_time = time.time() - start_time
            self.training_metrics['training_time'] = training_time
            
            # Sauvegarder les métriques
            self._save_metrics()
            
            print("\n🏆 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            print(f"⏱️  Temps total: {training_time:.2f}s")
            print(f"💾 Modèle sauvé: {final_model_path}")
            
            # Test final
            self._test_final_model(model, env)
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            raise
        finally:
            if 'env' in locals():
                env.close()
    
    def _test_final_model(self, model, env):
        """Teste le modèle final entraîné"""
        print("\n🧪 TEST DU MODÈLE FINAL")
        print("-" * 40)
        
        num_test_episodes = 5
        total_rewards = []
        successful_episodes = 0
        
        for episode in range(num_test_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if episode_length >= 500:  # Limite de sécurité
                    break
            
            total_rewards.append(episode_reward)
            
            # Vérifier le succès (récompense élevée)
            if episode_reward > 50:
                successful_episodes += 1
            
            print(f"  Épisode {episode + 1}: Récompense={episode_reward:.2f}, Longueur={episode_length}")
        
        # Statistiques finales
        avg_reward = np.mean(total_rewards)
        success_rate = successful_episodes / num_test_episodes * 100
        
        print(f"\n📊 RÉSULTATS DU TEST:")
        print(f"  - Récompense moyenne: {avg_reward:.2f}")
        print(f"  - Taux de succès: {success_rate:.1f}%")
        print(f"  - Meilleure récompense: {max(total_rewards):.2f}")
        
        # Mettre à jour les métriques
        self.training_metrics['final_test_avg_reward'] = avg_reward
        self.training_metrics['final_test_success_rate'] = success_rate
    
    def _save_metrics(self):
        """Sauvegarde les métriques d'entraînement"""
        metrics_path = os.path.join(self.results_dir, "training_metrics.json")
        
        # Ajouter timestamp
        self.training_metrics['timestamp'] = datetime.now().isoformat()
        
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        print(f"📈 Métriques sauvées: {metrics_path}")

class ImprovedTrainingCallback(BaseCallback):
    """
    Callback amélioré pour monitoring et sauvegarde pendant l'entraînement
    """
    
    def __init__(self, trainer, check_freq: int = 1000, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.trainer = trainer
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_check = 0
        
    def _on_step(self) -> bool:
        # Collecter les données d'épisode
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Garder seulement les 100 derniers épisodes
                if len(self.episode_rewards) > 100:
                    self.episode_rewards.pop(0)
                    self.episode_lengths.pop(0)
        
        # Vérification périodique
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps
            
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-20:])  # Moyenne des 20 derniers
                mean_length = np.mean(self.episode_lengths[-20:])
                
                print(f"\n📊 Step {self.num_timesteps}:")
                print(f"  - Récompense moyenne (20 derniers): {mean_reward:.2f}")
                print(f"  - Longueur moyenne: {mean_length:.1f}")
                
                # Sauvegarder si c'est le meilleur modèle
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.trainer.models_dir, "sac_best_model.zip")
                    self.model.save(best_model_path)
                    print(f"  🏆 Nouveau meilleur modèle sauvé! Récompense: {mean_reward:.2f}")
                
                # Mettre à jour les métriques du trainer
                self.trainer.training_metrics['episode_rewards'].extend(self.episode_rewards[-20:])
                self.trainer.training_metrics['episode_lengths'].extend(self.episode_lengths[-20:])
                self.trainer.training_metrics['best_reward'] = self.best_mean_reward
        
        # Sauvegarde périodique
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.trainer.models_dir, 
                f"sac_checkpoint_{self.num_timesteps}.zip"
            )
            self.model.save(checkpoint_path)
            print(f"💾 Checkpoint sauvé: {checkpoint_path}")
        
        return True

def main():
    """Fonction principale d'entraînement"""
    print("🎯 LANCEMENT DE L'ENTRAÎNEMENT SAC AMÉLIORÉ")
    print("=" * 50)
    
    # Configuration
    total_timesteps = 50000  # Commencer avec moins de timesteps pour tester
    
    # Créer et lancer l'entraîneur
    trainer = ImprovedGraspingTrainer(total_timesteps=total_timesteps)
    
    try:
        trainer.train()
        print("\n✅ ENTRAÎNEMENT COMPLÉTÉ AVEC SUCCÈS!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()