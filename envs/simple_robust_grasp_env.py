#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT DE GRASPING SIMPLIFIÉ ET ROBUSTE
================================================

Environnement inspiré du travail fonctionnel du collègue, simplifié pour éviter :
- La stagnation des rewards
- Les erreurs NaN/Inf
- Les vitesses excessives
- La complexité inutile du curriculum learning

Version professionnelle mais simple et fonctionnelle.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os
from typing import Dict, Tuple, Optional
import warnings
import tempfile

warnings.filterwarnings("ignore")

class SimpleRobustGraspEnv(gym.Env):
    """
    Environnement de grasping simplifié et robuste
    Basé sur le code fonctionnel du collègue
    """
    
    def __init__(self, 
                model_path: str = None, 
                render_mode: str = "rgb_array",
                eval_mode: bool = False):
        
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.eval_mode = eval_mode
        self.model_path = model_path or self._create_default_scene()
        
        # Charger le modèle MuJoCo
        self._load_model()
        
        # Identifier les composants du robot
        self._setup_robot_components()
        
        # Configuration des espaces
        self._setup_spaces()
        
        # Variables d'état
        self._initialize_state()
        
        print("✅ SimpleRobustGraspEnv initialisé avec succès!")
    
    def _create_default_scene(self) -> str:
        """Créer une scène par défaut si aucun modèle n'est fourni"""
        scene_xml = '''<?xml version="1.0" encoding="utf-8"?>
    +<mujoco model="simple_grasp">
    <compiler angle="radian"/>
    <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
    
    <asset>
        <material name="table_mat" rgba="0.8 0.6 0.4 1"/>
        <material name="cube_mat" rgba="0.4 0.6 0.8 1"/>
        <material name="hand_mat" rgba="0.7 0.7 0.7 1"/>
    </asset>
    
    <worldbody>
        <!-- Table -->
        <body name="table" pos="0 0 0.4">
            <geom type="box" size="0.5 0.5 0.05" material="table_mat"
                condim="3" contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
        </body>
        
        <!-- Robot base -->
        <body name="robot_base" pos="0 0 0.5">
            <!-- Right arm (simplified) -->
            <body name="right_shoulder" pos="0 -0.2 0.3">
                <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.03 0.1" rgba="0.7 0.7 0.7 1"/>
                
                <body name="right_elbow" pos="0 0 -0.2">
                    <joint name="right_elbow_pitch" type="hinge" axis="0 1 0" range="0 2.27"/>
                    <geom type="capsule" size="0.025 0.08" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="right_wrist" pos="0 0 -0.15">
                        <joint name="right_wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <joint name="right_wrist_roll" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                        <geom type="box" size="0.02 0.03 0.02" rgba="0.5 0.5 0.5 1"/>
                        
                        <!-- Hand with simplified fingers -->
                        <body name="right_hand_index_1_link" pos="0.04 0 0">
                            <joint name="right_hand_index_1_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                            <geom name="right_hand_index_1_geom" type="box" size="0.015 0.008 0.025" rgba="0.8 0.6 0.4 1"/>
                            
                            <body name="right_hand_index_2_link" pos="0.025 0 0">
                                <joint name="right_hand_index_2_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="right_hand_index_2_geom" type="box" size="0.012 0.006 0.02" rgba="0.8 0.6 0.4 1"/>
                            </body>
                        </body>
                        
                        <body name="right_hand_thumb_2_link" pos="0.02 0.025 0">
                            <joint name="right_hand_thumb_1_joint" type="hinge" axis="1 0 0" range="-0.5 1.2"/>
                            <joint name="right_hand_thumb_2_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                            <geom name="right_hand_thumb_2_geom" type="box" size="0.015 0.008 0.025" rgba="0.8 0.6 0.4 1"/>
                        </body>
                        
                        <body name="right_hand_middle_1_link" pos="0.04 -0.015 0">
                            <joint name="right_hand_middle_1_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                            <geom name="right_hand_middle_1_geom" type="box" size="0.015 0.008 0.025" rgba="0.8 0.6 0.4 1"/>
                            
                            <body name="right_hand_middle_2_link" pos="0.025 0 0">
                                <joint name="right_hand_middle_2_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="right_hand_middle_2_geom" type="box" size="0.012 0.006 0.02" rgba="0.8 0.6 0.4 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Cube -->
        <body name="cube" pos="0.18 0 0.55">
            <joint name="cube:joint" type="free"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" 
                material="cube_mat" condim="3" contype="1" conaffinity="1" 
                friction="0.8 0.005 0.0001"/>
        </body>
        
        <!-- Camera -->
        <camera name="main_cam" pos="0.8 0 1.0" xyaxes="1 0 0 0 0 1"/>
    </worldbody>
    
    <actuator>
        <!-- Arm actuators -->
        <motor name="right_shoulder_pitch_motor" joint="right_shoulder_pitch" gear="100"/>
        <motor name="right_elbow_pitch_motor" joint="right_elbow_pitch" gear="100"/>
        <motor name="right_wrist_pitch_motor" joint="right_wrist_pitch" gear="50"/>
        <motor name="right_wrist_roll_motor" joint="right_wrist_roll" gear="50"/>
        
        <!-- Finger actuators -->
        <motor name="right_hand_index_1_motor" joint="right_hand_index_1_joint" gear="20"/>
        <motor name="right_hand_index_2_motor" joint="right_hand_index_2_joint" gear="20"/>
        <motor name="right_hand_thumb_1_motor" joint="right_hand_thumb_1_joint" gear="20"/>
        <motor name="right_hand_thumb_2_motor" joint="right_hand_thumb_2_joint" gear="20"/>
        <motor name="right_hand_middle_1_motor" joint="right_hand_middle_1_joint" gear="20"/>
        <motor name="right_hand_middle_2_motor" joint="right_hand_middle_2_joint" gear="20"/>
    </actuator>
    +</mujoco>'''
        
        # Sauvegarder dans un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(scene_xml)
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
            self.model.opt.iterations = 100
            self.model.opt.tolerance = 1e-10
            
            # Renderer (optionnel pour les environnements headless)
            self.renderer = None
            try:
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
                print("✅ Renderer MuJoCo initialisé")
            except Exception as render_error:
                print(f"⚠️ Renderer non disponible (mode headless): {render_error}")
                print("   L'entraînement fonctionnera sans rendu visuel")
            
            print(f"✅ Modèle MuJoCo chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _setup_robot_components(self):
        """Identifier les composants du robot"""
        # Identifier les actuateurs du bras droit
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith("right_"):
                right_actuators.append(i)
        
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)
        
        # IDs des corps importants
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        print(f"✅ Composants identifiés: {len(self.right_actuator_ids)} actuateurs")
    
    def _setup_spaces(self):
        """Configuration des espaces d'action et d'observation"""
        # Espace d'action: contrôle des actuateurs du bras droit
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(len(self.right_actuator_ids),),
            dtype=np.float32
        )
        
        # Espace d'observation: positions, vitesses + positions 3D importantes
        obs_dim = self.model.nq + self.model.nv + 9  # +9 pour positions cube/main/relative
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"✅ Espaces configurés: Action {self.action_space.shape}, Obs {self.observation_space.shape}")
    
    def _initialize_state(self):
        """Initialiser les variables d'état"""
        self.current_step = 0
        self.max_steps = 500
        self.success_counter = 0
        self.freeze_timer = 0
        self.cube_initial_pos = None
    
    def reset(self, seed=None, options=None):
        """Reset de l'environnement"""
        super().reset(seed=seed)
        
        # Reset du modèle
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Forward pour stabiliser
        mujoco.mj_forward(self.model, self.data)
        
        # Position fixe du cube comme dans le code du collègue
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube:joint")
        if cube_joint_id >= 0:
            cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
            
            # Position fixe du cube (ajustée pour être plus stable)
            fixed_cube_pos = np.array([0.18, 0.0, 0.5])  # Plus haut pour éviter collision
            fixed_cube_quat = np.array([1, 0, 0, 0])
            
            self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = fixed_cube_pos
            self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = fixed_cube_quat
            
            self.cube_initial_pos = fixed_cube_pos.copy()
        
        # Stabiliser la simulation
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Step principal - inspiré du code du collègue"""
        # Validation des actions
        action = np.array(action, dtype=np.float32)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros_like(action)
        
        action = np.clip(action, -1.0, 1.0)
        
        # Split action: bras (premiers 4-7) + doigts (reste)
        arm_action = action[:4] if len(action) >= 4 else action
        finger_action = action[4:] if len(action) > 4 else np.array([])
        
        # Obtenir les positions importantes
        cube_pos = self._get_cube_position()
        palm_pos = self._get_hand_center()
        
        # Calcul des distances et contacts
        dist = np.linalg.norm(palm_pos - cube_pos)
        thumb_contact = self._is_touching("cube_geom", "right_hand_thumb_2_geom")
        index_contact = self._is_touching("cube_geom", "right_hand_index_1_geom")
        middle_contact = self._is_touching("cube_geom", "right_hand_middle_1_geom")
        num_contacts = sum([thumb_contact, index_contact, middle_contact])
        
        # Échelles adaptatives comme dans le code du collègue
        ARM_SCALE = 0.4 if dist > 0.08 else 0.2
        FINGER_SCALE = 0.7
        
        # Reset des contrôles
        self.data.ctrl[:] = 0.0
        
        # Application des actions avec échelles
        if len(arm_action) > 0:
            arm_actuators = self.right_actuator_ids[:len(arm_action)]
            self.data.ctrl[arm_actuators] = arm_action * ARM_SCALE
        
        if len(finger_action) > 0:
            finger_actuators = self.right_actuator_ids[len(arm_action):len(arm_action)+len(finger_action)]
            self.data.ctrl[finger_actuators] = finger_action * FINGER_SCALE
        
        # Assistance au grasping comme dans le code du collègue
        if dist < 0.06 and num_contacts >= 2:
            assist_strength = 0.5
            finger_actuators = self.right_actuator_ids[len(arm_action):]
            self.data.ctrl[finger_actuators] += assist_strength
            self.data.ctrl[finger_actuators] = np.clip(self.data.ctrl[finger_actuators], -1.0, 1.0)
            if not self.eval_mode:
                print("🤝 Assistance au grasping activée (≥2 doigts en contact)")
        
        # Simulation
        mujoco.mj_step(self.model, self.data)
        
        # Calculs pour la terminaison et récompense
        obs = self._get_obs()
        reward = self._compute_reward()
        
        self.current_step += 1
        
        # Conditions de terminaison ajustées (plus permissives)
        terminated = (
            dist > 0.8 or           # Distance plus permissive
            cube_pos[2] < -0.1 or   # Chute plus profonde
            cube_pos[2] > 1.5 or    # Plus haut autorisé
            self.current_step >= self.max_steps
        )
        
        truncated = False
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self):
        """Calcul de récompense simplifié basé sur le code du collègue"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_hand_center()
        
        dist = np.linalg.norm(palm_pos - cube_pos)
        cube_vel = np.linalg.norm(self._get_cube_velocity())
        
        # Compter les contacts
        fingers = ["right_hand_thumb_2_geom", "right_hand_index_1_geom", "right_hand_middle_1_geom"]
        touch_count = sum(self._is_touching(f, "cube_geom") for f in fingers)
        
        # Qualité du grasping (comme le collègue)
        if touch_count == 0:
            grasp_quality = -1.0
        elif touch_count == 1:
            grasp_quality = 0.1
        elif touch_count == 2:
            grasp_quality = 0.4
        else:  # 3+
            grasp_quality = 0.9 if cube_vel < 0.05 else 0.5
        
        # Composants de récompense (simplifiés du code du collègue)
        reward = 0
        reward += 5.0 / (1.0 + 20 * dist)  # Récompense de proximité
        reward += 2.0 if dist < 0.06 else 0  # Bonus proximité
        reward += 10.0 * grasp_quality  # Qualité du grasping
        reward -= 2.0 * min(1.0, cube_vel)  # Pénalité vitesse
        reward -= 0.005  # Pénalité temps
        
        if not self.eval_mode and self.current_step % 50 == 0:
            print(f"[step {self.current_step}] dist: {dist:.3f}, vel: {cube_vel:.3f}, "
                f"touches: {touch_count}, qualité: {grasp_quality:.2f}, reward: {reward:.2f}")
        
        return reward
    
    def _get_obs(self):
        """Observation simplifiée"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_hand_center()
        relative_pos = cube_pos - palm_pos
        
        # État de base (positions + vitesses)
        base_state = np.concatenate([self.data.qpos, self.data.qvel])
        
        # Observation complète
        obs = np.concatenate([base_state, cube_pos, palm_pos, relative_pos])
        
        # Vérification NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    def _get_cube_position(self):
        """Position du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id].copy()
        return np.zeros(3)
    
    def _get_cube_velocity(self):
        """Vitesse du cube"""
        if self.cube_body_id >= 0:
            return self.data.cvel[self.cube_body_id][:3].copy()
        return np.zeros(3)
    
    def _get_hand_center(self):
        """Position du centre de la main"""
        try:
            return self.data.body("right_hand_index_1_link").xpos.copy()
        except:
            # Fallback si le nom n'existe pas
            return np.array([0.0, 0.0, 0.5])
    
    def _is_touching(self, geom1, geom2):
        """Détection de contact entre deux géométries"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom1 in (name1, name2)) and (geom2 in (name1, name2)):
                return True
        return False
    
    def _get_info(self):
        """Informations de debug"""
        cube_pos = self._get_cube_position()
        palm_pos = self._get_hand_center()
        
        # Détection des contacts
        fingers = ["right_hand_thumb_2_geom", "right_hand_index_1_geom", "right_hand_middle_1_geom"]
        contacts = {finger: self._is_touching(finger, "cube_geom") for finger in fingers}
        
        # Lifting detection
        cube_lifted = False
        if self.cube_initial_pos is not None:
            lift_height = cube_pos[2] - self.cube_initial_pos[2]
            cube_lifted = lift_height > 0.03
        
        return {
            'step': self.current_step,
            'cube_position': cube_pos,
            'hand_position': palm_pos,
            'distance': np.linalg.norm(cube_pos - palm_pos),
            'contacts': contacts,
            'total_contacts': sum(contacts.values()),
            'cube_lifted': cube_lifted,
            'successful_grasp': sum(contacts.values()) >= 2 and cube_lifted
        }
    
    def render(self):
        """Rendu de l'environnement"""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera="main_cam")
                return self.renderer.render()
            except Exception as e:
                if not self.eval_mode:
                    print(f"⚠️ Erreur rendu: {e}")
                return None
        elif self.render_mode == "human":
            # Placeholder pour le rendu humain
            pass
        return None
    
    def close(self):
        """Fermeture propre"""
        if hasattr(self, 'temp_model_file'):
            try:
                os.unlink(self.temp_model_file)
            except:
                pass
        
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
            except:
                pass


# Test rapide
if __name__ == "__main__":
    print("🧪 Test de l'environnement simplifié...")
    
    env = SimpleRobustGraspEnv()
    
    obs, info = env.reset()
    print(f"✅ Reset OK - Obs shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, distance={info['distance']:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ Test réussi!")
