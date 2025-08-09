#!/usr/bin/env python3
"""
🤖 ENVIRONNEMENT ROBUSTE BASÉ SUR L'ANALYSE DU COLLÈGUE
======================================================

Implémentation des insights du collègue qui fonctionne:
1. Mouvement smooth vers le cube (scaling adaptatif)
2. Fixation de la palme au-dessus du cube  
3. Fermeture progressive des doigts

✅ Pure Reinforcement Learning - AUCUN CONTROL EXPLICITE
✅ Assistance progressive qui diminue vers l'autonomie
✅ Scaling adaptatif comme le collègue
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import tempfile
import time
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

class GraspPhase(Enum):
    """Phases naturelles du grasping"""
    PHASE_1_SMOOTH_APPROACH = "smooth_approach"  # Mouvement smooth vers cube
    PHASE_2_PALM_POSITIONING = "palm_positioning"  # Fixation palme au-dessus
    PHASE_3_FINGER_CLOSURE = "finger_closure"    # Fermeture doigts

class RobustSmoothGraspEnv(gym.Env):
    """
    🤖 Environnement robuste inspiré du code du collègue qui fonctionne
    
    Séquence naturelle :
    1. Mouvement SMOOTH vers le cube (avec scaling adaptatif)
    2. Fixation de la PALME au-dessus du cube  
    3. Fermeture progressive des DOIGTS
    
    ✅ Implemente TOUS les insights du collègue qui fonctionne
    ✅ Assistance progressive qui diminue
    ✅ Pure RL sans control explicite
    """
    
    def __init__(self, 
                 render_mode="rgb_array",
                 auto_phase_progression=True,
                 initial_assistance_level=0.6):
        
        super().__init__()
        
        self.render_mode = render_mode
        self.auto_phase_progression = auto_phase_progression
        self.initial_assistance_level = initial_assistance_level
        
        # Logger
        self._setup_logging()
        
        # Créer et charger le modèle
        self.model_path = self._create_robust_model()
        self._load_model()
        
        # Configuration des composants (comme le collègue)
        self._setup_robot_components()
        self._setup_spaces()
        
        # Variables d'état
        self._initialize_state()
        
        self.logger.info(f"🤖 Environnement robuste initialisé")
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger("RobustSmoothGrasp")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _create_robust_model(self):
        """Créer modèle MuJoCo robuste inspiré du collègue"""
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robust_smooth_grasp">
    <compiler angle="radian" meshdir="." texturedir="."/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" solver="PGS" iterations="50"/>
    
    <default>
        <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.1 0.05"/>
        <joint damping="0.1" stiffness="0"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <asset>
        <material name="table_mat" rgba="0.8 0.6 0.4 1" specular="0.3"/>
        <material name="cube_mat" rgba="0.2 0.6 0.8 1" specular="0.5"/>
        <material name="hand_mat" rgba="0.9 0.7 0.5 1" specular="0.3"/>
    </asset>
    
    <worldbody>
        <!-- Environment lighting -->
        <light name="top_light" pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 0.4">
            <geom type="box" size="0.6 0.6 0.05" material="table_mat" mass="50"/>
        </body>
        
        <!-- Robot arm and hand -->
        <body name="robot_base" pos="0 0 0.5">
            <!-- Shoulder -->
            <body name="shoulder" pos="0 -0.15 0.2">
                <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                <joint name="shoulder_tilt" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04 0.08" rgba="0.7 0.7 0.7 1"/>
                
                <!-- Upper arm -->
                <body name="upper_arm" pos="0 0 -0.15">
                    <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.5"/>
                    <geom type="capsule" size="0.03 0.1" rgba="0.6 0.6 0.6 1"/>
                    
                    <!-- Forearm -->
                    <body name="forearm" pos="0 0 -0.15">
                        <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                        <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom type="capsule" size="0.025 0.08" rgba="0.5 0.5 0.5 1"/>
                        
                        <!-- Hand avec sites pour contacts -->
                        <body name="right_hand_index_1_link" pos="0 0 -0.1">
                            <geom type="box" size="0.03 0.04 0.02" material="hand_mat"/>
                            
                            <!-- Thumb -->
                            <body name="right_hand_thumb_2_link" pos="0.02 0.03 0">
                                <joint name="right_hand_thumb_base" type="hinge" axis="1 0 0" range="-0.5 1.2"/>
                                <geom name="right_hand_thumb_2_geom" type="capsule" size="0.01 0.02" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="right_hand_thumb_tip" pos="0.015 0 0">
                                    <joint name="right_hand_thumb_tip" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="right_hand_thumb_tip_geom" type="capsule" size="0.008 0.015" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                            
                            <!-- Index finger -->
                            <body name="right_hand_index_2_link" pos="0.04 0.01 0">
                                <joint name="right_hand_index_base" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="right_hand_index_1_geom" type="capsule" size="0.01 0.025" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="right_hand_index_tip" pos="0.02 0 0">
                                    <joint name="right_hand_index_tip" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="right_hand_index_tip_geom" type="capsule" size="0.008 0.02" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                            
                            <!-- Middle finger -->
                            <body name="right_hand_middle_1_link" pos="0.04 -0.01 0">
                                <joint name="right_hand_middle_base" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="right_hand_middle_1_geom" type="capsule" size="0.01 0.025" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="right_hand_middle_tip" pos="0.02 0 0">
                                    <joint name="right_hand_middle_tip" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="right_hand_middle_tip_geom" type="capsule" size="0.008 0.02" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Target cube - Position FIXE comme le collègue -->
        <body name="cube" pos="0.18 0 0.5">
            <joint name="cube:joint" type="free"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" 
                  material="cube_mat" priority="1"/>
        </body>
        
        <!-- Camera views -->
        <camera name="main_cam" pos="0.8 0.5 0.8" xyaxes="1 0 0 0 1 1"/>
    </worldbody>
    
    <actuator>
        <!-- Arm actuators -->
        <motor name="right_shoulder_pan" joint="shoulder_pan" gear="100"/>
        <motor name="right_shoulder_tilt" joint="shoulder_tilt" gear="100"/>
        <motor name="right_elbow" joint="elbow" gear="80"/>
        <motor name="right_wrist_roll" joint="wrist_roll" gear="40"/>
        <motor name="right_wrist_pitch" joint="wrist_pitch" gear="40"/>
        
        <!-- Hand actuators -->
        <motor name="right_thumb_base" joint="right_hand_thumb_base" gear="20"/>
        <motor name="right_thumb_tip" joint="right_hand_thumb_tip" gear="15"/>
        <motor name="right_index_base" joint="right_hand_index_base" gear="20"/>
        <motor name="right_index_tip" joint="right_hand_index_tip" gear="15"/>
        <motor name="right_middle_base" joint="right_hand_middle_base" gear="20"/>
        <motor name="right_middle_tip" joint="right_hand_middle_tip" gear="15"/>
    </actuator>
</mujoco>'''
        
        # Sauvegarder dans fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(model_xml)
        temp_file.flush()
        temp_file.close()
        self.temp_model_file = temp_file.name
        return temp_file.name
    
    def _load_model(self):
        """Charger le modèle MuJoCo"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Configuration physique robuste
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-10
            
            # Renderer optionnel (headless compatible)
            self.renderer = None
            try:
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            except Exception as e:
                self.logger.warning(f"Renderer non disponible (mode headless): {e}")
            
            self.logger.info(f"✅ Modèle chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _setup_robot_components(self):
        """Setup robot components like colleague's code"""
        # Identifier les actuateurs comme le collègue
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None and name.startswith("right_"):
                right_actuators.append(i)
        
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)
        
        # Séparer bras et doigts comme le collègue
        self.arm_actuators = self.right_actuator_ids[:5]  # 5 premiers pour le bras
        self.finger_actuators = self.right_actuator_ids[5:]  # Reste pour les doigts
        
        # IDs des corps importants
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
        
        self.logger.info(f"✅ Composants: {len(self.arm_actuators)} bras, {len(self.finger_actuators)} doigts")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space: tous les actuateurs comme le collègue
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(len(self.right_actuator_ids),),
            dtype=np.float32
        )
        
        # Observation space riche pour RL
        obs_dim = (
            self.model.nq + self.model.nv +  # État robot
            6 +  # Positions cube et palme  
            4 +  # Distances et angles relatifs
            8 +  # Informations contacts et phase
            6    # Historique mouvement et smoothness
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.logger.info(f"✅ Espaces: Action {self.action_space.shape}, Obs {self.observation_space.shape}")
    
    def _initialize_state(self):
        """Initialiser variables d'état"""
        # Variables de base
        self.current_step = 0
        self.episode_count = 0
        self.max_steps = 500
        
        # Phase actuelle
        self.current_phase = GraspPhase.PHASE_1_SMOOTH_APPROACH
        self.phase_start_step = 0
        self.phase_history = []
        
        # Variables pour mouvements smooth
        self.previous_palm_pos = None
        self.velocity_history = []
        self.action_history = []
        
        # Variables d'assistance progressive
        self.current_assistance_level = self.initial_assistance_level
        self.assistance_decay_rate = 0.995
        
        # Métriques de performance
        self.best_distance_to_cube = float('inf')
        self.palm_stability_counter = 0
        self.contact_history = []
        self.smooth_movement_score = 0.0
        
        # Comme le collègue: position fixe du cube
        self.fixed_cube_pos = np.array([0.18, 0.0, 0.5])
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset physique
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.episode_count += 1
        
        # Reset des variables
        self._initialize_state()
        
        # Position du cube FIXE comme le collègue
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube:joint")
        if cube_joint_id >= 0:
            cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
            cube_quat = np.array([1, 0, 0, 0])
            self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = self.fixed_cube_pos
            self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = cube_quat
        
        # Position initiale du robot (légèrement variée)
        for i in range(min(5, self.model.nq)):
            if i < 3:  # Bras
                self.data.qpos[i] = np.random.uniform(-0.2, 0.2)
            else:  # Doigts ouverts
                self.data.qpos[i] = 0.0
        
        # Stabilisation
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.previous_palm_pos = self._get_palm_position()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """Step environment avec logique robuste du collègue"""
        # Validation et nettoyage des actions
        action = self._validate_and_process_actions(action)
        
        # Application des actions avec scaling adaptatif du collègue
        self._apply_actions_with_colleague_scaling(action)
        
        # Application de l'assistance progressive (pure RL, pas de control)
        self._apply_progressive_assistance()
        
        # Step physique
        mujoco.mj_step(self.model, self.data)
        
        # Calcul de l'observation
        observation = self._get_observation()
        
        # Calcul des rewards basé sur les phases
        reward = self._calculate_phase_based_reward()
        
        # Mise à jour des métriques
        self._update_metrics()
        
        # Progression automatique des phases
        if self.auto_phase_progression:
            self._check_phase_progression()
        
        # Conditions de terminaison
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _validate_and_process_actions(self, action):
        """Validation et traitement des actions comme le collègue"""
        action = np.array(action, dtype=np.float32)
        
        # Gestion NaN/Inf comme le collègue
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("Action NaN/Inf détectée, remplacement par zéros")
            action = np.zeros_like(action)
        
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        # Historique pour smoothness
        self.action_history.append(action.copy())
        if len(self.action_history) > 5:
            self.action_history.pop(0)
        
        return action
    
    def _apply_actions_with_colleague_scaling(self, action):
        """Application des actions avec scaling adaptatif du collègue"""
        # Reset des contrôles comme le collègue - CRUCIAL !
        self.data.ctrl[:] = 0.0
        
        # Obtenir positions
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(palm_pos - cube_pos)
        
        # Séparer actions bras/doigts comme le collègue
        arm_action = action[:len(self.arm_actuators)]
        finger_action = action[len(self.arm_actuators):]
        
        # SCALING ADAPTATIF du collègue - INSIGHT CLÉ !
        ARM_SCALE = 0.4 if dist > 0.08 else 0.2  # Mouvements lents quand proche
        FINGER_SCALE = 0.7
        
        # Application avec scaling
        self.data.ctrl[self.arm_actuators] = arm_action * ARM_SCALE
        if len(finger_action) > 0:
            self.data.ctrl[self.finger_actuators] = finger_action * FINGER_SCALE
    
    def _apply_progressive_assistance(self):
        """Assistance progressive qui diminue (pure RL, pas de control)"""
        if self.current_assistance_level <= 0.01:
            return
        
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(cube_pos - palm_pos)
        
        # Assistance contextuelle comme le collègue
        if self.current_phase == GraspPhase.PHASE_1_SMOOTH_APPROACH:
            # Aide pour mouvement smooth vers cube
            if dist > 0.1:
                direction = (cube_pos - palm_pos) / (dist + 1e-6)
                assistance = direction * self.current_assistance_level * 0.1
                # Appliquer aux 3 premiers actuateurs du bras
                for i in range(min(3, len(self.arm_actuators))):
                    if i < len(assistance):
                        self.data.ctrl[self.arm_actuators[i]] += assistance[i]
        
        elif self.current_phase == GraspPhase.PHASE_2_PALM_POSITIONING:
            # Aide pour stabiliser la palme
            if dist < 0.1:
                # Encourager la stabilité en réduisant les mouvements brusques
                stability_factor = self.current_assistance_level * 0.05
                self.data.ctrl[self.arm_actuators] *= (1.0 - stability_factor)
        
        elif self.current_phase == GraspPhase.PHASE_3_FINGER_CLOSURE:
            # Assistance pour fermeture comme le collègue
            num_contacts = self._get_contact_count()
            if dist < 0.06 and num_contacts >= 1:
                assist_strength = self.current_assistance_level * 0.5
                self.data.ctrl[self.finger_actuators] += assist_strength
                self.data.ctrl[self.finger_actuators] = np.clip(
                    self.data.ctrl[self.finger_actuators], -1.0, 1.0
                )
        
        # Diminution progressive de l'assistance
        self.current_assistance_level *= self.assistance_decay_rate
    
    def _calculate_phase_based_reward(self):
        """Calcul de reward basé sur la phase actuelle"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(cube_pos - palm_pos)
        
        total_reward = 0.0
        
        if self.current_phase == GraspPhase.PHASE_1_SMOOTH_APPROACH:
            total_reward += self._calculate_smooth_approach_reward(dist)
        elif self.current_phase == GraspPhase.PHASE_2_PALM_POSITIONING:
            total_reward += self._calculate_palm_positioning_reward(dist)
        elif self.current_phase == GraspPhase.PHASE_3_FINGER_CLOSURE:
            total_reward += self._calculate_finger_closure_reward(dist)
        
        # Rewards globaux comme le collègue
        total_reward += self._calculate_colleague_style_rewards(dist)
        
        # Pénalités
        total_reward -= self._calculate_penalties()
        
        return np.clip(total_reward, -20.0, 50.0)
    
    def _calculate_smooth_approach_reward(self, dist):
        """Reward pour mouvement smooth vers le cube"""
        reward = 0.0
        
        # Distance reward comme le collègue
        reward += 5.0 / (1.0 + 20 * dist)
        
        # Bonus progression
        if dist < self.best_distance_to_cube:
            self.best_distance_to_cube = dist
            reward += 2.0
        
        # Reward smoothness du mouvement
        if self.previous_palm_pos is not None:
            current_palm_pos = self._get_palm_position()
            velocity = np.linalg.norm(current_palm_pos - self.previous_palm_pos)
            # Récompenser vitesse modérée (pas trop lent, pas trop rapide)
            if 0.001 < velocity < 0.01:
                reward += 3.0
            elif velocity > 0.02:  # Trop rapide
                reward -= 1.0
        
        # Bonus proximité comme le collègue
        if dist < 0.08:
            reward += 5.0
            
        return reward
    
    def _calculate_palm_positioning_reward(self, dist):
        """Reward pour fixation de la palme"""
        reward = 0.0
        
        # Récompenser position proche
        if dist < 0.08:
            reward += 10.0
            
            # Bonus stabilité de la palme
            if self.previous_palm_pos is not None:
                palm_movement = np.linalg.norm(self._get_palm_position() - self.previous_palm_pos)
                if palm_movement < 0.005:  # Très stable
                    self.palm_stability_counter += 1
                    reward += 2.0
                else:
                    self.palm_stability_counter = 0
            
            # Bonus stabilité prolongée
            if self.palm_stability_counter > 10:
                reward += 5.0
        
        return reward
    
    def _calculate_finger_closure_reward(self, dist):
        """Reward pour fermeture des doigts"""
        reward = 0.0
        
        if dist < 0.08:
            # Récompenser les contacts comme le collègue
            contacts = self._get_detailed_contacts()
            num_contacts = len(contacts)
            
            if num_contacts > 0:
                reward += num_contacts * 5.0
                
                # Bonus qualité grasp comme le collègue
                if num_contacts == 1:
                    reward += 5.0
                elif num_contacts == 2:
                    reward += 15.0
                elif num_contacts >= 3:
                    reward += 25.0
                    
                # Stabilité du cube
                cube_vel = self._get_cube_velocity()
                if np.linalg.norm(cube_vel) < 0.05:
                    reward += 10.0
        
        return reward
    
    def _calculate_colleague_style_rewards(self, dist):
        """Rewards globaux dans le style du collègue"""
        reward = 0.0
        
        # Reward proximité du collègue
        if dist < 0.06:
            reward += 2.0
        
        # Pénalité vitesse cube comme le collègue
        cube_vel = self._get_cube_velocity()
        cube_speed = np.linalg.norm(cube_vel)
        reward -= 2.0 * min(1.0, cube_speed)
        
        # Pénalité temporelle comme le collègue
        reward -= 0.005
        
        return reward
    
    def _calculate_penalties(self):
        """Calcul des pénalités"""
        penalties = 0.0
        
        # Pénalité mouvements brusques (anti-smooth)
        if len(self.action_history) >= 2:
            action_diff = np.linalg.norm(self.action_history[-1] - self.action_history[-2])
            if action_diff > 0.5:
                penalties += action_diff * 2.0
        
        # Pénalité vitesses excessives
        max_joint_vel = np.max(np.abs(self.data.qvel))
        if max_joint_vel > 3.0:
            penalties += (max_joint_vel - 3.0)
        
        return penalties
    
    def _update_metrics(self):
        """Mise à jour des métriques de suivi"""
        # Historique vitesse pour smoothness
        if self.previous_palm_pos is not None:
            velocity = np.linalg.norm(self._get_palm_position() - self.previous_palm_pos)
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 20:
                self.velocity_history.pop(0)
        
        # Mise à jour position précédente
        self.previous_palm_pos = self._get_palm_position()
        
        # Historique contacts
        self.contact_history.append(self._get_contact_count())
        if len(self.contact_history) > 50:
            self.contact_history.pop(0)
    
    def _check_phase_progression(self):
        """Vérification progression automatique des phases"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(cube_pos - palm_pos)
        
        if self.current_phase == GraspPhase.PHASE_1_SMOOTH_APPROACH:
            # Passer à la phase 2 si proche et stable
            if dist < 0.08 and self.palm_stability_counter > 5:
                self._advance_to_phase(GraspPhase.PHASE_2_PALM_POSITIONING)
        
        elif self.current_phase == GraspPhase.PHASE_2_PALM_POSITIONING:
            # Passer à la phase 3 si palme stable
            if self.palm_stability_counter > 15:
                self._advance_to_phase(GraspPhase.PHASE_3_FINGER_CLOSURE)
        
        # Phase 3 reste jusqu'à la fin de l'épisode
    
    def _advance_to_phase(self, new_phase):
        """Avancer vers une nouvelle phase"""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_start_step = self.current_step
        self.phase_history.append({
            'from': old_phase.value,
            'to': new_phase.value,
            'step': self.current_step
        })
        
        self.logger.info(f"🎯 Phase transition: {old_phase.value} → {new_phase.value} (step {self.current_step})")
    
    def _get_observation(self):
        """Observation riche pour l'agent RL"""
        # État du robot
        robot_state = np.concatenate([self.data.qpos, self.data.qvel])
        
        # Positions
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        relative_pos = cube_pos - palm_pos
        
        # Distance et angle
        dist = np.linalg.norm(relative_pos)
        spatial_info = np.array([dist, relative_pos[0], relative_pos[1], relative_pos[2]])
        
        # Informations contacts et phase
        num_contacts = self._get_contact_count()
        phase_encoding = [
            1.0 if self.current_phase == GraspPhase.PHASE_1_SMOOTH_APPROACH else 0.0,
            1.0 if self.current_phase == GraspPhase.PHASE_2_PALM_POSITIONING else 0.0,
            1.0 if self.current_phase == GraspPhase.PHASE_3_FINGER_CLOSURE else 0.0,
            self.current_assistance_level,
            float(num_contacts),
            float(self.palm_stability_counter),
            float(self.current_step - self.phase_start_step),
            float(len(self.phase_history))
        ]
        
        # Informations mouvement et smoothness
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        velocity_std = np.std(self.velocity_history) if len(self.velocity_history) > 1 else 0.0
        recent_contacts = np.mean(self.contact_history[-5:]) if self.contact_history else 0.0
        movement_info = np.array([
            avg_velocity,
            velocity_std,
            recent_contacts,
            float(self.palm_stability_counter > 10),
            self.best_distance_to_cube,
            float(self.episode_count)
        ])
        
        # Observation complète
        observation = np.concatenate([
            robot_state,
            np.array([cube_pos[0], cube_pos[1], cube_pos[2]]),
            np.array([palm_pos[0], palm_pos[1], palm_pos[2]]),
            spatial_info,
            phase_encoding,
            movement_info
        ])
        
        # Nettoyage final
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation.astype(np.float32)
    
    def _get_info(self):
        """Informations détaillées"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        
        return {
            'step': self.current_step,
            'episode': self.episode_count,
            'current_phase': self.current_phase.value,
            'phase_step': self.current_step - self.phase_start_step,
            'distance': float(np.linalg.norm(cube_pos - palm_pos)),
            'contact_count': self._get_contact_count(),
            'palm_stability': self.palm_stability_counter,
            'assistance_level': float(self.current_assistance_level),
            'phase_history': self.phase_history.copy(),
            'avg_velocity': float(np.mean(self.velocity_history)) if self.velocity_history else 0.0,
            'smoothness_score': self._calculate_smoothness_score()
        }
    
    def _calculate_smoothness_score(self):
        """Score de smoothness du mouvement"""
        if len(self.velocity_history) < 5:
            return 0.0
        
        # Smoothness = faible variance de vitesse + vitesse modérée
        vel_std = np.std(self.velocity_history)
        vel_mean = np.mean(self.velocity_history)
        
        # Score idéal: vitesse modérée avec faible variance
        ideal_velocity = 0.005
        velocity_score = max(0, 1.0 - abs(vel_mean - ideal_velocity) * 100)
        variance_score = max(0, 1.0 - vel_std * 1000)
        
        return float((velocity_score + variance_score) / 2.0)
    
    # Méthodes utilitaires robustes
    
    def _get_cube_position(self):
        """Position du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id].copy()
        return self.fixed_cube_pos
    
    def _get_palm_position(self):
        """Position de la palme"""
        if self.palm_body_id >= 0:
            return self.data.xpos[self.palm_body_id].copy()
        return np.zeros(3)
    
    def _get_cube_velocity(self):
        """Vitesse du cube"""
        if self.cube_body_id >= 0:
            return self.data.cvel[self.cube_body_id][:3].copy()
        return np.zeros(3)
    
    def _get_contact_count(self):
        """Nombre de contacts comme le collègue"""
        return len(self._get_detailed_contacts())
    
    def _get_detailed_contacts(self):
        """Contacts détaillés comme le collègue"""
        contacts = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Vérifier contact cube-doigt
            if (geom1_name and geom2_name and
                (('cube' in geom1_name and 'right_hand' in geom2_name) or
                 ('cube' in geom2_name and 'right_hand' in geom1_name))):
                
                contacts.append({
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'force': np.linalg.norm(contact.force)
                })
        
        return contacts
    
    def _check_termination(self):
        """Conditions de terminaison"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_palm_position()
        dist = np.linalg.norm(cube_pos - palm_pos)
        
        # Conditions comme le collègue mais plus permissives
        if dist > 0.6:
            return True
        if cube_pos[2] < 0.3:
            return True
        if cube_pos[2] > 1.2:
            return True
        
        return False
    
    def render(self):
        """Rendu"""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera="main_cam")
                return self.renderer.render()
            except Exception as e:
                return None
        return None
    
    def close(self):
        """Fermeture"""
        if hasattr(self, 'temp_model_file'):
            try:
                import os
                os.unlink(self.temp_model_file)
            except:
                pass
        
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
            except:
                pass

# Test de l'environnement
if __name__ == "__main__":
    print("🤖 Test environnement robuste basé sur l'analyse du collègue...")
    
    env = RobustSmoothGraspEnv()
    obs, info = env.reset()
    
    print(f"✅ Environnement créé")
    print(f"   Phase initiale: {info['current_phase']}")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Assistance initiale: {info['assistance_level']:.2f}")
    
    # Test quelques steps
    total_reward = 0
    for i in range(30):
        action = env.action_space.sample() * 0.2  # Actions douces
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}: phase={info['current_phase']}, "
                  f"distance={info['distance']:.3f}, "
                  f"contacts={info['contact_count']}, "
                  f"smoothness={info['smoothness_score']:.2f}, "
                  f"assistance={info['assistance_level']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"✅ Test réussi! Reward total: {total_reward:.2f}")
    print(f"   Transitions de phase: {len(info['phase_history'])}")
    
    env.close()
    print("✅ Environnement fermé")