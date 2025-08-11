"""
🤖 ENVIRONNEMENT GRASPING OPTIMISÉ - INSPIRÉ DU COLLÈGUE
========================================================

Version optimisée qui intègre les meilleures pratiques du collègue
avec corrections des problèmes de stagnation identifiés.

✅ SOLUTIONS APPLIQUÉES:
- Position cube plus accessible [0.15, 0.0, 0.04] 
- Reset contrôles systématique
- Scaling adaptatif efficace
- Système rewards motivant
- Assistance grasping intelligente
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import tempfile
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import os

class OptimizedGraspEnv1(gym.Env):
    """
    🤖 Environnement de grasping optimisé - Version qui fonctionne!
    
    Inspiré des bonnes pratiques du collègue:
    - Reset contrôles: self.data.ctrl[:] = 0.0 (CRITIQUE!)
    - Scaling adaptatif: ARM_SCALE selon distance
    - Position cube accessible: [0.15, 0.0, 0.04]
    - Assistance grasp: aide quand >= 2 doigts touchent
    - Rewards bien calibrés pour convergence
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 render_mode: str = "rgb_array",
                 max_episode_steps: int = 500):
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Modèle XML optimisé avec fallback robuste
        if model_path is None:
            # Essayer les modèles dans l'ordre de préférence
            model_candidates = [
                "results/g1_combined_working.xml",
                "results/g1_combined.xml",
                "results/g1_combined_balanced.xml"
            ]
            
            model_path = None
            for candidate in model_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    self.logger.info(f"✅ Modèle trouvé: {candidate}")
                    break
            
            if model_path is None:
                raise FileNotFoundError("❌ Aucun modèle XML trouvé dans results/")
            
        self.model_path = model_path
        
        # Charger modèle MuJoCo avec fallback
        model_loaded = False
        for attempt, candidate in enumerate([model_path] + [
            "results/g1_combined.xml", 
            "results/g1_combined_balanced.xml"
        ]):
            if not os.path.exists(candidate):
                continue
                
            try:
                self.model = mujoco.MjModel.from_xml_path(candidate)
                self.data = mujoco.MjData(self.model)
                self.model_path = candidate
                self.logger.info(f"✅ Modèle chargé: {candidate}")
                model_loaded = True
                break
            except Exception as e:
                self.logger.warning(f"⚠️ Échec chargement {candidate}: {e}")
                continue
        
        if not model_loaded:
            raise RuntimeError("❌ Impossible de charger un modèle XML valide")
        
        # Configuration renderer
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        
        # ACTUATORS: Focus sur main droite seulement (comme le collègue)
        self._setup_actuators()
        
        # Spaces
        self._setup_spaces()
        
        # Variables d'état
        self.reset_episode_vars()
        
        self.logger.info(f"🤖 Environnement optimisé initialisé - {len(self.right_actuator_ids)} actuators")
    
    def _setup_actuators(self):
        """Configuration des actuators selon approche du collègue"""
        
        # Trouver actuators main droite
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None and "right_" in name:
                right_actuators.append(i)
        
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)
        
        # Séparer bras et doigts
        self.arm_actuators = []
        self.finger_actuators = []
        
        for actuator_id in self.right_actuator_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if any(joint in name for joint in ["shoulder", "elbow", "wrist"]):
                self.arm_actuators.append(actuator_id)
            else:
                self.finger_actuators.append(actuator_id)
        
        self.arm_actuators = np.array(self.arm_actuators)
        self.finger_actuators = np.array(self.finger_actuators)
        
        self.logger.info(f"Actuators bras: {len(self.arm_actuators)}, doigts: {len(self.finger_actuators)}")
    
    def _setup_spaces(self):
        """Configuration des espaces d'action et observation"""
        
        # Action space: tous les actuators main droite
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.right_actuator_ids),),
            dtype=np.float32
        )
        
        # Observation space: qpos + qvel + cube_pos + palm_pos + relative_pos
        obs_dim = self.model.nq + self.model.nv + 9  # +9 pour positions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset_episode_vars(self):
        """Reset variables d'épisode"""
        self.current_step = 0
        self.success_counter = 0
        self.best_distance = float('inf')
        self.total_reward = 0.0
    
    def reset(self, seed=None, options=None):
        """Reset environnement avec position cube optimisée"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset variables
        self.reset_episode_vars()
        
        # POSITION CUBE OPTIMISÉE (plus proche et accessible)
        try:
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            if cube_joint_id >= 0:
                cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
                
                # Position de base plus accessible
                fixed_cube_pos = np.array([0.15, 0.0, 0.04])
                
                # Appliquer position
                start = cube_qpos_addr
                end = min(cube_qpos_addr + 3, len(self.data.qpos))
                self.data.qpos[start:end] = fixed_cube_pos[:end-start]
                
                # Orientation
                if cube_qpos_addr + 7 <= len(self.data.qpos):
                    fixed_cube_quat = np.array([1, 0, 0, 0])
                    start = cube_qpos_addr + 3
                    end = cube_qpos_addr + 7
                    self.data.qpos[start:end] = fixed_cube_quat
                    
        except Exception as e:
            self.logger.warning(f"Erreur positionnement cube: {e}")
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        """Step avec approche optimisée du collègue"""
        
        # Validation action
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        
        # Séparation bras/doigts
        n_arm = len(self.arm_actuators)
        arm_action = action[:n_arm] if n_arm > 0 else np.array([])
        finger_action = action[n_arm:] if len(action) > n_arm else np.array([])
        
        # Calcul positions et distances
        positions = self._get_positions()
        dist = positions['palm_to_cube_dist']
        
        # SCALING ADAPTATIF (méthode du collègue qui marche!)
        ARM_SCALE = 0.4 if dist > 0.08 else 0.2
        FINGER_SCALE = 0.7
        
        # RESET CONTRÔLES (CRITIQUE pour éviter accumulation!)
        self.data.ctrl[:] = 0.0
        
        # Application actions avec scaling
        if len(self.arm_actuators) > 0 and len(arm_action) > 0:
            self.data.ctrl[self.arm_actuators] = arm_action * ARM_SCALE
        
        if len(self.finger_actuators) > 0 and len(finger_action) > 0:
            self.data.ctrl[self.finger_actuators] = finger_action * FINGER_SCALE
        
        # ASSISTANCE GRASPING (comme le collègue)
        self._apply_grasp_assistance(positions)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Calcul reward et termination
        obs = self._get_obs()
        reward = self._compute_reward(positions)
        terminated = self._check_termination(positions)
        
        # Mise à jour état
        self.current_step += 1
        self.total_reward += reward
        
        # Update best distance
        if dist < self.best_distance:
            self.best_distance = dist
        
        # Info
        info = {
            'distance': dist,
            'contact_count': positions['contact_count'],
            'cube_velocity': positions['cube_velocity'],
            'best_distance': self.best_distance,
            'total_reward': self.total_reward
        }
        
        return obs, reward, terminated, False, info
    
    def _get_positions(self):
        """Calcul positions et métriques clés"""
        
        # Positions
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = self.data.xpos[cube_id].copy()
        
        # Palm position (utiliser index comme référence)
        try:
            palm_pos = self.data.body("right_hand_index_1_link").xpos.copy()
        except:
            # Fallback si nom différent
            palm_pos = np.array([0.0, 0.0, 0.0])
        
        # Distances
        palm_to_cube_dist = np.linalg.norm(palm_pos - cube_pos)
        
        # Vélocité cube
        cube_velocity = np.linalg.norm(self.data.cvel[cube_id]) if cube_id >= 0 else 0.0
        
        # Contacts (méthode du collègue)
        contact_count = self._count_finger_contacts()
        
        return {
            'cube_pos': cube_pos,
            'palm_pos': palm_pos,
            'palm_to_cube_dist': palm_to_cube_dist,
            'cube_velocity': cube_velocity,
            'contact_count': contact_count
        }
    
    def _count_finger_contacts(self):
        """Compte contacts doigts-cube (méthode du collègue)"""
        
        fingers = [
            "right_hand_thumb_2_geom",
            "right_hand_index_1_geom", 
            "right_hand_middle_1_geom"
        ]
        
        contact_count = 0
        for finger in fingers:
            if self._is_touching("cube_geom", finger):
                contact_count += 1
        
        return contact_count
    
    def _is_touching(self, geom1, geom2):
        """Détection contact entre géométries"""
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if (geom1 in (name1, name2)) and (geom2 in (name1, name2)):
                return True
        
        return False
    
    def _apply_grasp_assistance(self, positions):
        """Assistance grasping intelligente (du collègue)"""
        
        dist = positions['palm_to_cube_dist']
        contact_count = positions['contact_count']
        
        # Assistance quand >= 2 doigts touchent et proche
        if dist < 0.06 and contact_count >= 2:
            assist_strength = 0.5
            
            if len(self.finger_actuators) > 0:
                # Ajouter assistance fermeture
                self.data.ctrl[self.finger_actuators] += assist_strength
                self.data.ctrl[self.finger_actuators] = np.clip(
                    self.data.ctrl[self.finger_actuators], -1.0, 1.0
                )
                
                print(f"🤝 Assistance grasping activée! ({contact_count} doigts)")
    
    def _compute_reward(self, positions):
        """Système de reward optimisé (inspiré du collègue)"""
        
        dist = positions['palm_to_cube_dist']
        cube_vel = positions['cube_velocity']
        contact_count = positions['contact_count']
        
        # Grasp quality heuristic (du collègue)
        if contact_count == 0:
            grasp_quality = -1.0
        elif contact_count == 1:
            grasp_quality = 0.1
        elif contact_count == 2:
            grasp_quality = 0.4
        else:  # 3+
            grasp_quality = 0.9 if cube_vel < 0.05 else 0.5
        
        # Composants reward (calibrés pour convergence)
        reward = 0.0
        
        # Reward distance (encourage approche)
        reward += 5.0 / (1.0 + 20 * dist)
        
        # Bonus proximité
        if dist < 0.06:
            reward += 2.0
        
        # Reward grasping
        reward += 10.0 * grasp_quality
        
        # Pénalité vélocité cube
        reward -= 2.0 * min(1.0, cube_vel)
        
        # Pénalité temps (légère)
        reward -= 0.005
        
        # Bonus succès (grasp stable)
        if contact_count >= 2 and dist < 0.05 and cube_vel < 0.02:
            reward += 20.0
            self.success_counter += 1
        
        return reward
    
    def _check_termination(self, positions):
        """Conditions de terminaison"""
        
        dist = positions['palm_to_cube_dist']
        cube_pos = positions['cube_pos']
        
        # Terminaison si:
        terminated = (
            dist > 0.5 or                        # Trop loin
            cube_pos[2] < 0.01 or               # Cube tombé
            cube_pos[2] > 1.0 or                # Cube trop haut
            self.current_step >= self.max_episode_steps  # Max steps
        )
        
        return terminated
    
    def _get_obs(self):
        """Observation selon format du collègue"""
        
        # Positions de base
        base_state = np.concatenate([self.data.qpos, self.data.qvel])
        
        # Positions spécifiques
        positions = self._get_positions()
        cube_pos = positions['cube_pos']
        palm_pos = positions['palm_pos']
        relative_pos = cube_pos - palm_pos
        
        # Observation complète
        obs = np.concatenate([base_state, cube_pos, palm_pos, relative_pos])
        obs = obs.astype(np.float32)
        
        # Ajustement taille
        expected_dim = self.observation_space.shape[0]
        fixed_obs = np.zeros(expected_dim, dtype=np.float32)
        fixed_obs[:min(expected_dim, obs.shape[0])] = obs[:min(expected_dim, obs.shape[0])]
        
        return fixed_obs
    
    def render(self):
        """Rendu visuel"""
        
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
        return None
    
    def close(self):
        """Nettoyage ressources"""
        
        if hasattr(self, 'renderer'):
            self.renderer.close()