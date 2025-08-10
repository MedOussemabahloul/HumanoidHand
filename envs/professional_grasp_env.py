"""
Environnement professionnel pour l'apprentissage de saisie robotique
Architecture modulaire et configurable
"""

import numpy as np
import gymnasium as gym
import mujoco
from typing import Dict, Any, Tuple, Optional
import logging
from config import Config

class ProfessionalGraspEnv(gym.Env):
    """
    Environnement professionnel pour l'apprentissage de saisie
    
    Caractéristiques:
    - Architecture modulaire
    - Configuration centralisée
    - Gestion d'erreurs robuste
    - Logging intégré
    - Extensible pour curriculum learning
    """
    
    def __init__(self, config: Optional[Config] = None, eval_mode: bool = False):
        """
        Initialise l'environnement
        
        Args:
            config: Configuration personnalisée (utilise DEFAULT_CONFIG si None)
            eval_mode: Mode évaluation (comportement déterministe)
        """
        super().__init__()
        
        # Configuration
        if config is None:
            from config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        
        self.config = config
        self.eval_mode = eval_mode
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup environnement système
        self.config.setup_environment()
        
        # Charger le modèle MuJoCo
        self._load_model()
        
        # Identifier les composants du robot
        self._identify_robot_components()
        
        # Configurer les espaces d'action et observation
        self._setup_spaces()
        
        # Initialiser les compteurs
        self.reset_counters()
        
        self.logger.info("✅ Environnement professionnel initialisé")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure le système de logging"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.system.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Charge le modèle MuJoCo avec gestion d'erreurs"""
        model_path = self.config.get_model_path()
        
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            self.logger.info(f"Modèle chargé: {model_path}")
            self.logger.info(f"Timestep: {self.model.opt.timestep}")
            self.logger.info(f"DOFs: {self.model.nv}, Actuateurs: {self.model.nu}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {model_path}")
    
    def _identify_robot_components(self):
        """Identifie les composants du robot (actuateurs, corps, etc.)"""
        
        # Identifier les actuateurs droits
        self.right_actuator_ids = []
        self.arm_actuator_ids = []
        self.finger_actuator_ids = []
        
        for i in range(self.model.nu):
            actuator_name = self.model.actuator(i).name
            
            if 'right' in actuator_name:
                self.right_actuator_ids.append(i)
                
                # Classifier par type
                if any(joint in actuator_name for joint in ['shoulder', 'elbow', 'wrist']):
                    self.arm_actuator_ids.append(i)
                elif any(finger in actuator_name for finger in ['index', 'middle', 'ring', 'thumb']):
                    self.finger_actuator_ids.append(i)
        
        # Identifier les corps importants
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.right_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        
        # Validation
        if self.cube_id < 0:
            raise RuntimeError("Corps 'cube' non trouvé dans le modèle")
        
        if self.right_hand_id < 0:
            self.logger.warning("Corps 'right_wrist_yaw_link' non trouvé, utilisation de fallback")
            # Fallback: utiliser le premier corps de doigt droit
            self.right_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_index_0")
        
        self.logger.info(f"Actuateurs droits: {len(self.right_actuator_ids)} identifiés")
        self.logger.info(f"Cube ID: {self.cube_id}, Main droite ID: {self.right_hand_id}")
    
    def _setup_spaces(self):
        """Configure les espaces d'action et observation"""
        
        # Espace d'action: tous les actuateurs droits
        n_actions = len(self.right_actuator_ids)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_actions,),
            dtype=np.float32
        )
        
        # Espace d'observation: positions + optionnellement vitesses
        obs_components = []
        
        # Position du cube (3D)
        obs_components.append(3)
        
        # Positions des joints du bras droit (7 DOFs)
        obs_components.append(7)
        
        # Positions des doigts droits (8 DOFs)
        obs_components.append(8)
        
        # Optionnel: vitesses
        if self.config.environment.include_velocities:
            obs_components.extend([3, 7, 8])  # Vitesses correspondantes
        
        obs_size = sum(obs_components)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.logger.info(f"Action space: {self.action_space.shape}")
        self.logger.info(f"Observation space: {self.observation_space.shape}")
    
    def reset_counters(self):
        """Reset des compteurs internes"""
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.best_distance = np.inf
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset de l'environnement
        
        Args:
            seed: Graine aléatoire
            options: Options additionnelles
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reset de la simulation MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale du cube avec variation
        base_pos = np.array(self.config.environment.cube_initial_pos)
        
        if not self.eval_mode:
            # Ajouter du bruit en mode entraînement
            noise = np.random.uniform(
                -self.config.environment.cube_position_noise,
                self.config.environment.cube_position_noise,
                3
            )
            cube_pos = base_pos + noise
        else:
            # Position fixe en mode évaluation
            cube_pos = base_pos
        
        # Appliquer la position
        self.data.qpos[0:3] = cube_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Quaternion identité
        
        # Reset des contrôles
        self.data.ctrl[:] = 0.0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset des compteurs
        self.step_count = 0
        self.episode_count += 1
        self.total_reward = 0.0
        
        # Observation initiale
        observation = self._get_observation()
        
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une action dans l'environnement
        
        Args:
            action: Action à exécuter (normalisée entre -1 et 1)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        
        # Validation de l'action
        action = np.clip(action, -1.0, 1.0)
        
        # Reset des contrôles (important pour la stabilité)
        self.data.ctrl[:] = 0.0
        
        # Appliquer l'action avec échelle adaptative
        self._apply_action(action)
        
        # Simulation step
        mujoco.mj_step(self.model, self.data)
        
        # Calcul du reward
        reward = self._calculate_reward()
        
        # Observation
        observation = self._get_observation()
        
        # Conditions de terminaison
        terminated = self._check_termination()
        truncated = self.step_count >= self.config.environment.max_episode_steps
        
        # Mise à jour des compteurs
        self.step_count += 1
        self.total_reward += reward
        
        # Info minimal pour Stable-Baselines3
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Applique l'action aux actuateurs avec échelle adaptative"""
        
        # Calculer la distance pour adaptation
        cube_pos = self.data.xpos[self.cube_id]
        hand_pos = self.data.xpos[self.right_hand_id] if self.right_hand_id >= 0 else np.zeros(3)
        distance = np.linalg.norm(cube_pos - hand_pos)
        
        # Facteur d'adaptation (plus proche = mouvements plus doux)
        distance_factor = min(1.0, distance / self.config.robot.distance_adaptation_threshold)
        
        # Séparer actions bras/doigts
        arm_actions = action[:len(self.arm_actuator_ids)]
        finger_actions = action[len(self.arm_actuator_ids):len(self.arm_actuator_ids) + len(self.finger_actuator_ids)]
        
        # Appliquer aux actuateurs de bras
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if i < len(arm_actions):
                scaled_action = (arm_actions[i] * 
                               self.config.robot.arm_action_scale * 
                               distance_factor)
                self.data.ctrl[actuator_id] = scaled_action
        
        # Appliquer aux actuateurs de doigts
        for i, actuator_id in enumerate(self.finger_actuator_ids):
            if i < len(finger_actions):
                scaled_action = finger_actions[i] * self.config.robot.finger_action_scale
                self.data.ctrl[actuator_id] = scaled_action
    
    def _calculate_reward(self) -> float:
        """Calcule le reward basé sur la configuration"""
        
        cube_pos = self.data.xpos[self.cube_id]
        
        # Distance main-cube
        if self.right_hand_id >= 0:
            hand_pos = self.data.xpos[self.right_hand_id]
            distance = np.linalg.norm(cube_pos - hand_pos)
            
            # Mise à jour de la meilleure distance
            self.best_distance = min(self.best_distance, distance)
        else:
            distance = 2.0  # Fallback
        
        # Composantes du reward
        rewards = {}
        
        # 1. Reward de distance (progressif)
        if distance < self.config.reward.close_distance_threshold:
            rewards['distance'] = self.config.reward.close_distance_bonus
        elif distance < self.config.reward.medium_distance_threshold:
            rewards['distance'] = self.config.reward.medium_distance_bonus
        else:
            rewards['distance'] = -distance * self.config.reward.distance_weight
        
        # 2. Reward de contact
        rewards['contact'] = min(self.data.ncon * self.config.reward.contact_weight, 20.0)
        
        # 3. Reward de hauteur (éviter que le cube tombe)
        rewards['height'] = max(0, cube_pos[2] - 0.9) * self.config.reward.height_weight
        
        # 4. Reward de stabilité (pénaliser vitesses excessives)
        cube_vel = np.linalg.norm(self.data.cvel[self.cube_id][:3])
        rewards['stability'] = -min(cube_vel, 5.0) * self.config.reward.stability_weight
        
        # Total
        total_reward = sum(rewards.values())
        
        # Stocker pour info
        self.last_reward_components = rewards
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Génère l'observation actuelle"""
        
        observations = []
        
        # Position du cube
        cube_pos = self.data.xpos[self.cube_id][:3]
        observations.append(cube_pos)
        
        # Positions des joints du bras droit (DOFs 13-19)
        right_arm_qpos = self.data.qpos[13:20]
        observations.append(right_arm_qpos)
        
        # Positions des doigts droits (DOFs 28-35)
        right_finger_qpos = self.data.qpos[28:36]
        observations.append(right_finger_qpos)
        
        # Optionnel: vitesses
        if self.config.environment.include_velocities:
            cube_vel = self.data.cvel[self.cube_id][:3]
            right_arm_qvel = self.data.qvel[13:20]
            right_finger_qvel = self.data.qvel[28:36]
            
            observations.extend([cube_vel, right_arm_qvel, right_finger_qvel])
        
        # Concaténer toutes les observations
        obs = np.concatenate(observations)
        
        # Sécurité: éliminer NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalisation optionnelle
        if self.config.environment.normalize_observations:
            obs = np.clip(obs, -10.0, 10.0)  # Clipping conservateur
        
        return obs.astype(np.float32)
    
    def _check_termination(self) -> bool:
        """Vérifie les conditions de terminaison"""
        
        # Terminaison si instabilité détectée
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
            np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            self.logger.warning("Terminaison: Instabilité numérique détectée")
            return True
        
        # Terminaison si cube tombe trop bas
        cube_pos = self.data.xpos[self.cube_id]
        if cube_pos[2] < 0.3:
            self.logger.debug("Terminaison: Cube tombé")
            return True
        
        # Succès: cube saisi et stable
        if (self.data.ncon > 3 and 
            self.right_hand_id >= 0 and
            np.linalg.norm(self.data.xpos[self.cube_id] - self.data.xpos[self.right_hand_id]) < 0.1):
            self.logger.info("Succès: Cube saisi!")
            return True
        
        return False
    
    def _get_step_info(self, reward: float) -> Dict[str, Any]:
        """Génère les informations de debug pour ce step"""
        
        cube_pos = self.data.xpos[self.cube_id]
        distance = (np.linalg.norm(cube_pos - self.data.xpos[self.right_hand_id]) 
                   if self.right_hand_id >= 0 else 2.0)
        
        return {
            "step": self.step_count,
            "episode": self.episode_count,
            "reward": reward,
            "total_reward": self.total_reward,
            "distance": distance,
            "best_distance": self.best_distance,
            "cube_pos": cube_pos.copy(),
            "contacts": self.data.ncon,
            "reward_components": getattr(self, 'last_reward_components', {}),
            "stable": not (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)))
        }
    
    def get_success_rate(self) -> float:
        """Retourne le taux de succès (pour évaluation)"""
        # À implémenter selon vos critères de succès
        return 0.0
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Rendu (désactivé pour performance)"""
        if mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None
    
    def close(self):
        """Fermeture propre de l'environnement"""
        self.logger.info("Fermeture de l'environnement")
        # Cleanup si nécessaire

def make_professional_env(config: Optional[Config] = None, eval_mode: bool = False) -> ProfessionalGraspEnv:
    """
    Factory function pour créer l'environnement
    
    Args:
        config: Configuration personnalisée
        eval_mode: Mode évaluation
        
    Returns:
        Instance de ProfessionalGraspEnv
    """
    return ProfessionalGraspEnv(config=config, eval_mode=eval_mode)
