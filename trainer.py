"""
Framework d'entraînement professionnel
Support pour différents algorithmes et stratégies
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config import Config
from envs.professional_grasp_env import make_professional_env

class TrainingMonitor(BaseCallback):
    """Monitoring professionnel de l'entraînement"""
    
    def __init__(self, config: Config, log_freq: int = 1000):
        super().__init__()
        self.config = config
        self.log_freq = log_freq
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.stats_file = Path(config.system.results_dir) / "training_progress.json"
        
        # Setup logging
        self.train_logger = logging.getLogger("TrainingMonitor")
        self.train_logger.setLevel(logging.INFO)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._log_progress()
        return True
    
    def _log_progress(self):
        """Log des progrès d'entraînement"""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
        
        # Test rapide de performance
        env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
        test_reward = self._quick_performance_test(env)
        
        if test_reward > self.best_reward:
            self.best_reward = test_reward
            self.train_logger.info(f"🎯 Nouveau record: {test_reward:.2f}")
        
        # Stats
        stats = {
            "step": self.n_calls,
            "elapsed_minutes": elapsed / 60,
            "steps_per_sec": steps_per_sec,
            "test_reward": test_reward,
            "best_reward": self.best_reward,
            "timestamp": time.time()
        }
        
        # Sauvegarder
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.train_logger.info(f"📊 Step {self.n_calls} | Reward: {test_reward:.2f} | Best: {self.best_reward:.2f} | Speed: {steps_per_sec:.1f} steps/s")
    
    def _quick_performance_test(self, env, n_steps: int = 10) -> float:
        """Test rapide de performance"""
        try:
            obs, _ = env.reset()
            total_reward = 0
            
            for _ in range(n_steps):
                action = env.action_space.sample()
                obs, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                if term or trunc:
                    break
            
            return total_reward / n_steps
        except Exception as e:
            self.train_logger.warning(f"Erreur test performance: {e}")
            return -100.0

class ProfessionalTrainer:
    """
    Trainer professionnel pour l'apprentissage de saisie
    
    Fonctionnalités:
    - Support multi-algorithmes (TD3, SAC, PPO)
    - Monitoring avancé
    - Sauvegarde automatique
    - Curriculum learning optionnel
    - Évaluation périodique
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self.results_dir = Path(config.system.results_dir)
        self.logs_dir = Path(config.system.logs_dir)
        
        # Créer les dossiers
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 Trainer professionnel initialisé")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup du logging"""
        logger = logging.getLogger("ProfessionalTrainer")
        logger.setLevel(getattr(logging, self.config.system.log_level))
        
        if not logger.handlers:
            # File handler
            log_file = self.config.system.logs_dir + "/training.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def create_environment(self, eval_mode: bool = False):
        """Crée l'environnement d'entraînement"""
        env = make_professional_env(config=self.config, eval_mode=eval_mode)
        
        # Wrapper Monitor pour logging
        log_dir = self.logs_dir / "monitor"
        log_dir.mkdir(exist_ok=True)
        
        env = Monitor(env, str(log_dir))
        
        return env
    
    def create_model(self, env):
        """Crée le modèle d'apprentissage selon la configuration"""
        
        if self.config.training.algorithm == "TD3":
            return self._create_td3_model(env)
        elif self.config.training.algorithm == "SAC":
            return self._create_sac_model(env)
        elif self.config.training.algorithm == "PPO":
            return self._create_ppo_model(env)
        else:
            raise ValueError(f"Algorithme non supporté: {self.config.training.algorithm}")
    
    def _create_td3_model(self, env):
        """Crée un modèle TD3"""
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=self.config.training.action_noise_sigma * np.ones(env.action_space.shape[0])
        )
        
        return TD3(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            buffer_size=self.config.training.buffer_size,
            gamma=self.config.training.gamma,
            tau=self.config.training.tau,
            policy_delay=self.config.training.policy_delay,
            target_policy_noise=self.config.training.target_policy_noise,
            target_noise_clip=self.config.training.target_noise_clip,
            verbose=1
        )
    
    def _create_sac_model(self, env):
        """Crée un modèle SAC"""
        return SAC(
            'MlpPolicy',
            env,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            buffer_size=self.config.training.buffer_size,
            gamma=self.config.training.gamma,
            tau=self.config.training.tau,
            verbose=1
        )
    
    def _create_ppo_model(self, env):
        """Crée un modèle PPO"""
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            gamma=self.config.training.gamma,
            verbose=1
        )
    
    def train(self, model_name: str = "professional_model") -> str:
        """
        Lance l'entraînement complet
        
        Args:
            model_name: Nom du modèle à sauvegarder
            
        Returns:
            Chemin du modèle final
        """
        
        self.logger.info("🎓 DÉMARRAGE ENTRAÎNEMENT PROFESSIONNEL")
        self.logger.info(f"Algorithme: {self.config.training.algorithm}")
        self.logger.info(f"Total timesteps: {self.config.training.total_timesteps}")
        
        # Créer environnement
        env = self.create_environment()
        
        # Test initial
        self._initial_environment_test(env)
        
        # Créer modèle
        model = self.create_model(env)
        
        # Callbacks
        monitor = TrainingMonitor(self.config, log_freq=self.config.training.eval_freq)
        
        # Entraînement par phases
        save_freq = self.config.training.save_freq
        total_steps = self.config.training.total_timesteps
        n_phases = max(1, total_steps // save_freq)
        
        for phase in range(n_phases):
            phase_steps = min(save_freq, total_steps - phase * save_freq)
            
            self.logger.info(f"🔄 Phase {phase+1}/{n_phases} - {phase_steps} steps")
            
            # Entraînement
            model.learn(
                total_timesteps=phase_steps,
                callback=monitor,
                reset_num_timesteps=False
            )
            
            # Sauvegarde intermédiaire
            phase_path = self.results_dir / f"{model_name}_phase_{phase+1}"
            model.save(str(phase_path))
            self.logger.info(f"💾 Phase {phase+1} sauvegardée")
        
        # Sauvegarde finale
        final_path = self.results_dir / f"{model_name}_final"
        model.save(str(final_path))
        
        # Évaluation finale
        self._final_evaluation(model, env)
        
        env.close()
        
        self.logger.info("🎉 ENTRAÎNEMENT TERMINÉ!")
        return str(final_path)
    
    def _initial_environment_test(self, env):
        """Test initial de l'environnement"""
        self.logger.info("🧪 Test initial de l'environnement...")
        
        obs, _ = env.reset()
        total_reward = 0
        
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if i % 5 == 0:
                self.logger.debug(f"Step {i}: reward={reward:.2f}")
        
        avg_reward = total_reward / 20
        self.logger.info(f"✅ Test initial - Reward moyen: {avg_reward:.2f}")
        
        if avg_reward < -100:
            self.logger.warning("⚠️ Rewards très négatifs - vérifiez la configuration")
    
    def _final_evaluation(self, model, env):
        """Évaluation finale du modèle"""
        self.logger.info("🎯 Évaluation finale...")
        
        eval_episodes = 10
        total_rewards = []
        
        for episode in range(eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            for _ in range(500):  # Max steps par épisode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                episode_reward += reward
                
                if term or trunc:
                    break
            
            total_rewards.append(episode_reward)
            self.logger.info(f"Épisode {episode+1}: {episode_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        self.logger.info(f"📊 ÉVALUATION FINALE:")
        self.logger.info(f"   Reward moyen: {avg_reward:.2f} ± {std_reward:.2f}")
        self.logger.info(f"   Meilleur épisode: {max(total_rewards):.2f}")
        
        # Sauvegarder résultats
        results = {
            "final_evaluation": {
                "episodes": eval_episodes,
                "mean_reward": avg_reward,
                "std_reward": std_reward,
                "max_reward": max(total_rewards),
                "all_rewards": total_rewards
            },
            "config": self.config.to_dict(),
            "timestamp": time.time()
        }
        
        with open(self.results_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)

def quick_train(timesteps: int = 50_000, algorithm: str = "TD3") -> str:
    """
    Entraînement rapide avec configuration par défaut
    
    Args:
        timesteps: Nombre de timesteps
        algorithm: Algorithme à utiliser
        
    Returns:
        Chemin du modèle entraîné
    """
    
    # Configuration par défaut
    from config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG
    config.training.total_timesteps = timesteps
    config.training.algorithm = algorithm
    
    # Entraînement
    trainer = ProfessionalTrainer(config)
    return trainer.train(f"quick_{algorithm.lower()}_model")

def production_train(config_file: Optional[str] = None) -> str:
    """
    Entraînement de production avec configuration personnalisée
    
    Args:
        config_file: Chemin vers fichier de configuration JSON (optionnel)
        
    Returns:
        Chemin du modèle entraîné
    """
    
    # Charger configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        # TODO: Implémenter chargement config depuis JSON
        config = DEFAULT_CONFIG
    else:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Entraînement
    trainer = ProfessionalTrainer(config)
    return trainer.train("production_model")

if __name__ == "__main__":
    print("🚀 FRAMEWORK D'ENTRAÎNEMENT PROFESSIONNEL")
    print("=" * 50)
    
    # Test rapide
    print("🧪 Test du framework...")
    
    try:
        model_path = quick_train(timesteps=5_000, algorithm="TD3")
        print(f"✅ Test réussi! Modèle: {model_path}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()