"""
Configuration centralisée pour l'entraînement de saisie robotique
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class SimulationConfig:
    """Configuration des paramètres de simulation MuJoCo"""
    timestep: float = 0.005
    solver: str = "PGS"
    iterations: int = 100
    tolerance: float = 1e-8
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

@dataclass
class RobotConfig:
    """Configuration du robot et des actuateurs"""
    # Échelles d'action
    arm_action_scale: float = 0.6
    finger_action_scale: float = 0.4
    
    # Gains des actuateurs (kp, kv)
    arm_gains: Tuple[float, float] = (80.0, 20.0)
    finger_gains: Tuple[float, float] = (25.0, 10.0)
    
    # Limites de force
    arm_force_limit: float = 100.0
    finger_force_limit: float = 25.0
    
    # Facteur d'adaptation à la distance
    distance_adaptation_threshold: float = 0.3

@dataclass
class EnvironmentConfig:
    """Configuration de l'environnement d'entraînement"""
    # Position initiale du cube
    cube_initial_pos: Tuple[float, float, float] = (0.5, 0.2, 1.0)
    cube_position_noise: float = 0.05
    
    # Paramètres d'épisode
    max_episode_steps: int = 1000
    
    # Observation
    include_velocities: bool = False
    normalize_observations: bool = True

@dataclass
class RewardConfig:
    """Configuration des rewards"""
    # Poids des différentes composantes
    distance_weight: float = 10.0
    contact_weight: float = 2.0
    stability_weight: float = 1.0
    height_weight: float = 2.0
    
    # Seuils pour rewards progressifs
    close_distance_threshold: float = 0.1
    close_distance_bonus: float = 10.0
    
    medium_distance_threshold: float = 0.2
    medium_distance_bonus: float = 5.0
    
    # Pénalités
    far_distance_penalty: float = -10.0
    instability_penalty: float = -5.0

@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    # Algorithme
    algorithm: str = "TD3"  # TD3, SAC, PPO
    
    # Hyperparamètres TD3
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 500_000
    gamma: float = 0.99
    tau: float = 0.005
    
    # Bruit d'action
    action_noise_sigma: float = 0.25
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    
    # Entraînement
    total_timesteps: int = 100_000
    save_freq: int = 10_000
    eval_freq: int = 2_000
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: List[Dict] = None

@dataclass
class SystemConfig:
    """Configuration système"""
    # Rendu
    render_mode: str = "osmesa"  # osmesa, egl, none
    headless: bool = True
    
    # Logging
    log_level: str = "INFO"
    save_videos: bool = False
    video_freq: int = 10_000
    
    # Paths
    model_path: str = "/workspace/results/g1_combined_balanced.xml"
    results_dir: str = "/workspace/results"
    logs_dir: str = "/workspace/logs"

class Config:
    """Configuration principale - point d'entrée unique"""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.robot = RobotConfig()
        self.environment = EnvironmentConfig()
        self.reward = RewardConfig()
        self.training = TrainingConfig()
        self.system = SystemConfig()
        
        # Créer les dossiers nécessaires
        os.makedirs(self.system.results_dir, exist_ok=True)
        os.makedirs(self.system.logs_dir, exist_ok=True)
    
    def get_model_path(self) -> str:
        """Retourne le chemin du modèle XML"""
        return self.system.model_path
    
    def setup_environment(self):
        """Configure l'environnement système"""
        if self.system.headless:
            os.environ["MUJOCO_GL"] = self.system.render_mode
            os.environ["PYTHONWARNINGS"] = "ignore"
    
    def to_dict(self) -> Dict:
        """Convertit la config en dictionnaire pour sauvegarde"""
        return {
            "simulation": self.simulation.__dict__,
            "robot": self.robot.__dict__,
            "environment": self.environment.__dict__,
            "reward": self.reward.__dict__,
            "training": self.training.__dict__,
            "system": self.system.__dict__
        }

# Configuration par défaut
DEFAULT_CONFIG = Config()