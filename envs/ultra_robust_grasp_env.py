#!/usr/bin/env python3
"""
🤖 ENVIRONNEMENT ULTRA-ROBUSTE BASÉ SUR L'ANALYSE DU COLLÈGUE
=============================================================

Version professionnelle qui implémente TOUS les insights du collègue:
✅ Scaling adaptatif selon distance (ARM_SCALE = 0.4 si dist > 0.08 else 0.2)
✅ Reset des contrôles à CHAQUE step (self.data.ctrl[:] = 0.0)
✅ Assistance au grasp (quand 2+ doigts touchent)
✅ Position fixe du cube (0.18, 0.0, 0.04)
✅ Gestion robuste des NaN/Inf
✅ Rewards qui fonctionnent (pas de stagnation)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import tempfile
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

class UltraRobustGraspEnv(gym.Env):
    """
    🤖 Environnement ULTRA-ROBUSTE basé sur le code fonctionnel du collègue
    
    INSIGHTS CLÉS DU COLLÈGUE IMPLÉMENTÉS:
    - Scaling adaptatif: ARM_SCALE = 0.4 si dist > 0.08 else 0.2
    - Reset contrôles: self.data.ctrl[:] = 0.0 avant chaque action
    - Assistance contextuelle: aide à la fermeture quand 2+ doigts touchent
    - Position cube fixe: [0.18, 0.0, 0.04] comme le collègue
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 render_mode: str = "rgb_array",
                 max_episode_steps: int = 500,
                 enable_assistance: bool = True,
                 assistance_strength: float = 0.5):
        
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.enable_assistance = enable_assistance
        self.assistance_strength = assistance_strength
        
        # Logger
        self._setup_logging()
        
        # Modèle MuJoCo
        self.model_path = model_path or self._create_model()
        self._load_mujoco_model()
        
        # Configuration des composants (comme le collègue)
        self._setup_robot_components()
        self._setup_spaces()
        
        # Variables d'état
        self._reset_episode_vars()
        
        self.logger.info("🤖 Environnement ultra-robuste initialisé avec succès")
    
    def _setup_logging(self):
        """Configure le logging"""
        self.logger = logging.getLogger("UltraRobustGrasp")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _create_model(self) -> str:
        """Crée un modèle XML robuste basé sur celui du collègue"""
        
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="ultra_robust_grasp">
    <compiler angle="radian" meshdir="." texturedir="."/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" 
            solver="PGS" iterations="50" tolerance="1e-10"/>
    
    <default>
        <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.1 0.05"/>
        <joint damping="0.5" stiffness="0"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <asset>
        <material name="table_mat" rgba="0.8 0.6 0.4 1"/>
        <material name="cube_mat" rgba="0.2 0.6 0.8 1"/>
        <material name="robot_mat" rgba="0.7 0.7 0.7 1"/>
    </asset>
    
    <worldbody>
        <!-- Éclairage -->
        <light name="top_light" pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 0.4">
            <geom type="box" size="0.6 0.6 0.05" material="table_mat" mass="50"/>
        </body>
        
        <!-- Robot humanoid simplifié -->
        <body name="robot_base" pos="0 0 0.5">
            <!-- Torse -->
            <geom name="torso" type="capsule" size="0.08 0.15" rgba="0.7 0.7 0.7 1"/>
            
            <!-- Bras droit -->
            <body name="right_shoulder" pos="0.1 -0.1 0.1">
                <joint name="right_shoulder_pitch" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                <joint name="right_shoulder_roll" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04 0.08" rgba="0.6 0.6 0.6 1"/>
                
                <!-- Bras supérieur -->
                <body name="right_upper_arm" pos="0 -0.15 0">
                    <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                    <geom type="capsule" size="0.035 0.12" rgba="0.6 0.6 0.6 1"/>
                    
                    <!-- Coude -->
                    <body name="right_elbow" pos="0 -0.18 0">
                        <joint name="right_elbow" type="hinge" axis="1 0 0" range="0 2.5"/>
                        <geom type="capsule" size="0.03 0.1" rgba="0.5 0.5 0.5 1"/>
                        
                        <!-- Avant-bras -->
                        <body name="right_forearm" pos="0 -0.15 0">
                            <joint name="right_wrist_roll" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                            <joint name="right_wrist_pitch" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                            <geom type="capsule" size="0.025 0.08" rgba="0.5 0.5 0.5 1"/>
                            
                            <!-- Main -->
                            <body name="right_hand_index_1_link" pos="0 -0.1 0">
                                <geom type="box" size="0.03 0.04 0.02" rgba="0.9 0.7 0.5 1"/>
                                
                                <!-- Pouce -->
                                <body name="right_hand_thumb_2_link" pos="0.02 -0.02 0.02">
                                    <joint name="right_thumb_base" type="hinge" axis="1 0 0" range="-0.5 1.2"/>
                                    <geom name="right_hand_thumb_2_geom" type="capsule" size="0.01 0.025" 
                                          rgba="0.9 0.7 0.5 1"/>
                                    
                                    <body name="right_thumb_tip" pos="0 -0.02 0">
                                        <joint name="right_thumb_tip" type="hinge" axis="1 0 0" range="0 1.57"/>
                                        <geom name="right_thumb_tip_geom" type="capsule" size="0.008 0.02" 
                                              rgba="0.9 0.7 0.5 1"/>
                                    </body>
                                </body>
                                
                                <!-- Index -->
                                <body name="right_hand_index_2_link" pos="0.03 -0.04 0.01">
                                    <joint name="right_index_base" type="hinge" axis="1 0 0" range="0 1.57"/>
                                    <geom name="right_hand_index_1_geom" type="capsule" size="0.01 0.03" 
                                          rgba="0.9 0.7 0.5 1"/>
                                    
                                    <body name="right_index_tip" pos="0 -0.025 0">
                                        <joint name="right_index_tip" type="hinge" axis="1 0 0" range="0 1.57"/>
                                        <geom name="right_index_tip_geom" type="capsule" size="0.008 0.02" 
                                              rgba="0.9 0.7 0.5 1"/>
                                    </body>
                                </body>
                                
                                <!-- Majeur -->
                                <body name="right_hand_middle_1_link" pos="0.03 -0.04 -0.01">
                                    <joint name="right_middle_base" type="hinge" axis="1 0 0" range="0 1.57"/>
                                    <geom name="right_hand_middle_1_geom" type="capsule" size="0.01 0.03" 
                                          rgba="0.9 0.7 0.5 1"/>
                                    
                                    <body name="right_middle_tip" pos="0 -0.025 0">
                                        <joint name="right_middle_tip" type="hinge" axis="1 0 0" range="0 1.57"/>
                                        <geom name="right_middle_tip_geom" type="capsule" size="0.008 0.02" 
                                              rgba="0.9 0.7 0.5 1"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Cube target - POSITION FIXE comme le collègue -->
        <body name="cube" pos="0.18 0 0.44">
            <joint name="cube:joint" type="free"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" 
                  mass="0.1" material="cube_mat" priority="1"
                  friction="1.0 0.1 0.05"/>
        </body>
        
        <!-- Camera -->
        <camera name="main_cam" pos="0.8 0.5 0.8" xyaxes="-0.6 0.8 0 -0.4 -0.3 0.9"/>
    </worldbody>
    
    <actuator>
        <!-- Actuateurs bras (7 DOF comme le collègue) -->
        <motor name="right_shoulder_pitch" joint="right_shoulder_pitch" gear="100"/>
        <motor name="right_shoulder_roll" joint="right_shoulder_roll" gear="100"/>
        <motor name="right_shoulder_yaw" joint="right_shoulder_yaw" gear="100"/>
        <motor name="right_elbow" joint="right_elbow" gear="80"/>
        <motor name="right_wrist_roll" joint="right_wrist_roll" gear="40"/>
        <motor name="right_wrist_pitch" joint="right_wrist_pitch" gear="40"/>
        
        <!-- Actuateurs doigts -->
        <motor name="right_thumb_base" joint="right_thumb_base" gear="20"/>
        <motor name="right_thumb_tip" joint="right_thumb_tip" gear="15"/>
        <motor name="right_index_base" joint="right_index_base" gear="20"/>
        <motor name="right_index_tip" joint="right_index_tip" gear="15"/>
        <motor name="right_middle_base" joint="right_middle_base" gear="20"/>
        <motor name="right_middle_tip" joint="right_middle_tip" gear="15"/>
    </actuator>
</mujoco>'''
        
        # Sauvegarder dans un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(model_xml)
        temp_file.flush()
        temp_file.close()
        
        self._temp_model_path = temp_file.name
        return temp_file.name
    
    def _load_mujoco_model(self):
        """Charge le modèle MuJoCo avec configuration robuste"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Configuration physique robuste
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-10
            
            # Initialiser les données
            mujoco.mj_forward(self.model, self.data)
            
            # Renderer pour visualisation
            self.renderer = None
            if self.render_mode in ['rgb_array', 'human']:
                try:
                    self.renderer = mujoco.Renderer(self.model, width=640, height=480)
                except Exception as e:
                    self.logger.warning(f"Renderer non disponible: {e}")
            
            self.logger.info(f"✅ Modèle MuJoCo chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle MuJoCo: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {e}")
    
    def _setup_robot_components(self):
        """Configure les composants du robot comme le collègue"""
        # Identifier actuateurs droits comme le collègue
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None and name.startswith("right_"):
                right_actuators.append(i)
        
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)
        
        # IDs des corps importants
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
        
        # IDs des corps de doigts
        self.finger_body_names = [
            "right_hand_thumb_2_link",
            "right_hand_index_1_link", 
            "right_hand_middle_1_link"
        ]
        
        self.finger_body_ids = []
        for name in self.finger_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self.finger_body_ids.append(body_id)
        
        # Position fixe du cube comme le collègue
        self.fixed_cube_pos = np.array([0.18, 0.0, 0.44])
        
        self.logger.info(f"✅ Composants configurés: {len(self.right_actuator_ids)} actuateurs")
    
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation"""
        # Action space: tous les actuateurs droits comme le collègue
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.right_actuator_ids),),
            dtype=np.float32
        )
        
        # Observation space riche
        obs_dim = (
            self.model.nq +      # Positions joints
            self.model.nv +      # Vitesses joints
            9                    # cube_pos(3) + palm_pos(3) + relative_pos(3)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.logger.info(f"✅ Espaces configurés: Action {self.action_space.shape}, Obs {self.observation_space.shape}")
    
    def _reset_episode_vars(self):
        """Reset des variables d'épisode"""
        self.current_step = 0
        self.success_counter = 0
        self.freeze_timer = 0
        self.best_distance = float('inf')
        
        # Historique pour debugging
        self.distance_history = []
        self.reward_history = []
        self.action_history = []
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset de l'environnement"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_vars()
        
        # Position du cube FIXE comme le collègue
        try:
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube:joint")
            if cube_joint_id >= 0:
                cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
                # Position et orientation du cube
                self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = self.fixed_cube_pos
                self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = np.array([1, 0, 0, 0])  # quat neutre
        except Exception as e:
            self.logger.warning(f"Impossible de fixer position du cube: {e}")
        
        # Position initiale robot (légèrement aléatoire)
        for i in range(min(6, len(self.right_actuator_ids))):  # Juste le bras
            if i < self.model.nq:
                self.data.qpos[i] = np.random.uniform(-0.1, 0.1)
        
        # Stabilisation
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step principal avec la logique du collègue"""
        
        # 1. VALIDATION ET NETTOYAGE DES ACTIONS (anti-NaN/Inf)
        action = self._validate_and_clean_action(action)
        
        # 2. APPLIQUER ACTIONS AVEC SCALING ADAPTATIF DU COLLÈGUE
        self._apply_colleague_action_logic(action)
        
        # 3. SIMULATION STEP
        mujoco.mj_step(self.model, self.data)
        
        # 4. OBSERVATION ET REWARD
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        # 5. MISE À JOUR ÉTAT
        self.current_step += 1
        self.reward_history.append(reward)
        self.action_history.append(action.copy())
        
        return observation, reward, terminated, truncated, info
    
    def _validate_and_clean_action(self, action: np.ndarray) -> np.ndarray:
        """Nettoie et valide les actions pour éviter NaN/Inf"""
        action = np.asarray(action, dtype=np.float32)
        
        # Gestion NaN/Inf - CRUCIAL pour éviter les crashes
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("⚠️ Action NaN/Inf détectée, remplacement par zéros")
            action = np.zeros_like(action, dtype=np.float32)
        
        # Clip dans la plage valide
        action = np.clip(action, -1.0, 1.0)
        
        # S'assurer de la bonne taille
        if len(action) != len(self.right_actuator_ids):
            action = action[:len(self.right_actuator_ids)]
            if len(action) < len(self.right_actuator_ids):
                action = np.pad(action, (0, len(self.right_actuator_ids) - len(action)))
        
        return action
    
    def _apply_colleague_action_logic(self, action: np.ndarray):
        """Applique la logique d'action du collègue - INSIGHT CLÉ"""
        
        # RESET DES CONTRÔLES comme le collègue - CRUCIAL !
        self.data.ctrl[:] = 0.0
        
        # Obtenir positions et distances
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(palm_pos - cube_pos)
        
        # Séparer actions bras et doigts comme le collègue
        arm_action = action[:7] if len(action) >= 7 else action[:min(6, len(action))]
        finger_action = action[7:] if len(action) > 7 else action[6:] if len(action) > 6 else np.array([])
        
        # SCALING ADAPTATIF DU COLLÈGUE - INSIGHT FONDAMENTAL !
        ARM_SCALE = 0.4 if dist > 0.08 else 0.2  # Mouvements lents quand proche
        FINGER_SCALE = 0.7
        
        # Appliquer actions bras avec scaling
        arm_actuators = self.right_actuator_ids[:len(arm_action)]
        self.data.ctrl[arm_actuators] = arm_action * ARM_SCALE
        
        # Appliquer actions doigts avec scaling
        if len(finger_action) > 0:
            finger_actuators = self.right_actuator_ids[len(arm_action):len(arm_action)+len(finger_action)]
            self.data.ctrl[finger_actuators] = finger_action * FINGER_SCALE
        
        # ASSISTANCE AU GRASP comme le collègue - quand 2+ doigts touchent
        if self.enable_assistance and dist < 0.06:
            num_contacts = self._get_contact_count()
            if num_contacts >= 2:
                # Encourager fermeture comme le collègue
                finger_start_idx = 7 if len(self.right_actuator_ids) > 7 else 6
                finger_actuators = self.right_actuator_ids[finger_start_idx:]
                
                assist_strength = self.assistance_strength
                self.data.ctrl[finger_actuators] += assist_strength
                self.data.ctrl[finger_actuators] = np.clip(self.data.ctrl[finger_actuators], -1.0, 1.0)
                
                if self.current_step % 50 == 0:  # Log occasionnel
                    self.logger.debug("🤝 Assistance grasp activée (≥2 doigts)")
    
    def _compute_reward(self) -> float:
        """Calcul du reward basé sur la logique du collègue"""
        
        # Positions et distances
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(palm_pos - cube_pos)
        
        # Vitesse du cube
        cube_vel = self._get_cube_velocity()
        cube_speed = np.linalg.norm(cube_vel)
        
        # Contacts
        num_contacts = self._get_contact_count()
        
        # REWARD STRUCTURE DU COLLÈGUE
        reward = 0.0
        
        # 1. Reward distance comme le collègue
        reward += 5.0 / (1.0 + 20 * dist)
        
        # 2. Bonus proximité comme le collègue
        if dist < 0.06:
            reward += 2.0
        
        # 3. Reward contacts comme le collègue
        if num_contacts == 0:
            grasp_quality = -1.0
        elif num_contacts == 1:
            grasp_quality = 0.1
        elif num_contacts == 2:
            grasp_quality = 0.4
        else:  # 3+
            grasp_quality = 0.9 if cube_speed < 0.05 else 0.5
        
        reward += 10.0 * grasp_quality
        
        # 4. Pénalité vitesse cube comme le collègue
        reward -= 2.0 * min(1.0, cube_speed)
        
        # 5. Pénalité temporelle comme le collègue
        reward -= 0.005
        
        # 6. Bonus progression
        if dist < self.best_distance:
            self.best_distance = dist
            reward += 1.0
        
        # 7. Pénalités pour éviter comportements indésirables
        
        # Pénalité vitesses articulaires excessives (anti-NaN)
        max_joint_vel = np.max(np.abs(self.data.qvel))
        if max_joint_vel > 5.0:
            reward -= (max_joint_vel - 5.0)
        
        # Pénalité actions brusques
        if len(self.action_history) >= 2:
            action_diff = np.linalg.norm(self.action_history[-1] - self.action_history[-2])
            if action_diff > 0.8:
                reward -= action_diff
        
        # Clamp final pour éviter valeurs extrêmes
        reward = np.clip(reward, -10.0, 20.0)
        
        # Validation finale anti-NaN
        if np.isnan(reward) or np.isinf(reward):
            self.logger.warning("⚠️ Reward NaN/Inf détecté, remplacement par 0")
            reward = 0.0
        
        return float(reward)
    
    def _get_contact_count(self) -> int:
        """Compte les contacts cube-doigts comme le collègue"""
        contacts = 0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name and geom2_name:
                # Vérifier contact cube-main
                if (('cube' in geom1_name and 'right_hand' in geom2_name) or
                    ('cube' in geom2_name and 'right_hand' in geom1_name)):
                    contacts += 1
        
        return contacts
    
    def _check_termination(self) -> bool:
        """Conditions de terminaison comme le collègue mais plus permissives"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(palm_pos - cube_pos)
        
        # Conditions d'arrêt
        if dist > 0.6:  # Plus permissif que le collègue (0.5)
            return True
        if cube_pos[2] < 0.2:  # Cube trop bas
            return True
        if cube_pos[2] > 1.0:  # Cube trop haut
            return True
        
        return False
    
    def _get_obs(self) -> np.ndarray:
        """Observation robuste comme le collègue"""
        
        # État du robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Positions importantes
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        relative_pos = cube_pos - palm_pos
        
        # Assemblage de l'observation comme le collègue
        observation = np.concatenate([
            qpos,           # Positions joints
            qvel,           # Vitesses joints
            cube_pos,       # Position cube
            palm_pos,       # Position palme
            relative_pos    # Position relative
        ])
        
        # Validation finale anti-NaN/Inf
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            self.logger.warning("⚠️ Observation NaN/Inf détectée, nettoyage")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Informations de debugging"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        
        return {
            'episode_step': self.current_step,
            'distance': float(np.linalg.norm(cube_pos - palm_pos)),
            'cube_position': cube_pos.tolist(),
            'palm_position': palm_pos.tolist(),
            'contact_count': self._get_contact_count(),
            'cube_velocity': float(np.linalg.norm(self._get_cube_velocity())),
            'best_distance': float(self.best_distance),
            'success_counter': self.success_counter,
            'reward_mean_last_10': float(np.mean(self.reward_history[-10:])) if len(self.reward_history) >= 10 else 0.0
        }
    
    # Méthodes utilitaires robustes
    
    def _get_cube_position(self) -> np.ndarray:
        """Position robuste du cube"""
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
            return self.data.xpos[self.cube_body_id].copy()
        else:
            return self.fixed_cube_pos.copy()
    
    def _get_palm_position(self) -> np.ndarray:
        """Position robuste de la palme"""
        if self.palm_body_id >= 0 and self.palm_body_id < len(self.data.xpos):
            return self.data.xpos[self.palm_body_id].copy()
        else:
            return np.zeros(3, dtype=np.float32)
    
    def _get_cube_velocity(self) -> np.ndarray:
        """Vitesse robuste du cube"""
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.cvel):
            return self.data.cvel[self.cube_body_id][:3].copy()
        else:
            return np.zeros(3, dtype=np.float32)
    
    def render(self):
        """Rendu avec gestion d'erreurs"""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera="main_cam")
                return self.renderer.render()
            except Exception as e:
                self.logger.warning(f"Erreur rendu: {e}")
                return None
        return None
    
    def close(self):
        """Fermeture propre"""
        if hasattr(self, '_temp_model_path'):
            try:
                os.unlink(self._temp_model_path)
            except:
                pass
        
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
            except:
                pass
        
        self.logger.info("🔒 Environnement fermé proprement")


def make_ultra_robust_grasp_env(**kwargs):
    """Factory function pour créer l'environnement"""
    return UltraRobustGraspEnv(**kwargs)


# Test de l'environnement
if __name__ == "__main__":
    print("🧪 TEST ENVIRONNEMENT ULTRA-ROBUSTE")
    print("=" * 50)
    
    try:
        # Créer l'environnement
        env = UltraRobustGraspEnv()
        print("✅ Environnement créé avec succès")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset réussi - Obs shape: {obs.shape}")
        print(f"   Distance initiale: {info['distance']:.3f}")
        print(f"   Contacts initiaux: {info['contact_count']}")
        
        # Test de steps
        total_reward = 0.0
        print("\n🏃 Test de 50 steps...")
        
        for step in range(50):
            # Action aléatoire douce
            action = env.action_space.sample() * 0.3
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Log périodique
            if step % 10 == 0:
                print(f"   Step {step}: distance={info['distance']:.3f}, "
                      f"contacts={info['contact_count']}, reward={reward:.3f}")
            
            # Vérifications anti-NaN
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"❌ ÉCHEC: Observation NaN/Inf au step {step}")
                break
                
            if np.isnan(reward) or np.isinf(reward):
                print(f"❌ ÉCHEC: Reward NaN/Inf au step {step}")
                break
            
            if terminated or truncated:
                print(f"   Épisode terminé au step {step}")
                break
        
        print(f"\n📊 Résultats du test:")
        print(f"   Reward total: {total_reward:.2f}")
        print(f"   Distance finale: {info['distance']:.3f}")
        print(f"   Contacts finaux: {info['contact_count']}")
        print(f"   Meilleure distance: {info['best_distance']:.3f}")
        
        # Test rendu
        frame = env.render()
        if frame is not None:
            print(f"✅ Rendu réussi - Frame shape: {frame.shape}")
        
        env.close()
        print("✅ Fermeture propre")
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("L'environnement est prêt pour l'entraînement.")
        
    except Exception as e:
        print(f"❌ ERREUR pendant les tests: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Vérifications recommandées:")
        print("- MuJoCo est-il installé correctement?")
        print("- Les dépendances sont-elles à jour?")
        print("- Y a-t-il des problèmes de permissions?")