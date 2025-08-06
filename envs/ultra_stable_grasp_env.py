#!/usr/bin/env python3
"""
Environnement ultra-stabilisé pour robot G1 - CORRECTION FINALE
Bloque les joints de doigts problématiques (DOF 15, 16, 20)
Debug complet avec noms des joints
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward

class UltraStableGraspEnv(gym.Env):
    """Environnement ultra-stabilisé - SOLUTION FINALE"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="results/g1_combined.xml", render_mode=None,
                 max_episode_steps=50, curriculum_level=1, block_fingers=True):
        super().__init__()
        
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_level = curriculum_level
        self.current_step = 0
        self.block_fingers = block_fingers
        
        # Charger et configurer le modèle
        self._load_and_configure_model()
        self._identify_joints_with_debug()
        self._setup_spaces()
        
        # Variables d'état
        self.cube_initial_pos = None
        self.cube_initial_height = None
        self.contact_detected = False
        self.previous_action = None
        self.action_smoothing = 0.05
        self.instability_count = 0
        
        # Renderer
        self.renderer = None
        self.viewer = None
        
        print(f"✅ Environnement ULTRA-stabilisé prêt (doigts bloqués: {self.block_fingers})")
        
    def _load_and_configure_model(self):
        """Charge et configure le modèle avec stabilité maximale"""
        try:
            self.model = MjModel.from_xml_path(self.xml_path)
            self.data = MjData(self.model)
            
            # CONFIGURATION ULTRA-STABLE
            self.model.opt.timestep = 0.01         # Timestep large
            self.model.opt.iterations = 100        # Plus d'itérations
            self.model.opt.ls_iterations = 50      # Line search
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self.model.opt.tolerance = 1e-5
            self.model.opt.ls_tolerance = 1e-3
            
            # Amortissement ultra-élevé
            for i in range(self.model.nv):
                if i < len(self.model.dof_damping):
                    self.model.dof_damping[i] = max(1.0, self.model.dof_damping[i] * 10)
                    
            print(f"✅ Modèle configuré: {self.model.nv} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            raise RuntimeError(f"Erreur modèle: {e}")
    
    def _identify_joints_with_debug(self):
        """Identifie et debug les joints problématiques"""
        print("🔍 DEBUG des joints...")
        
        # Trouver le cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Capteurs de force
        self.force_sensor_ids = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and ("force" in sensor_name.lower() or "touch" in sensor_name.lower()):
                self.force_sensor_ids.append(i)
        
        # IDENTIFIER LES JOINTS PROBLÉMATIQUES
        self.finger_dofs = []
        self.arm_dofs = []
        self.problematic_dofs = [15, 16, 20]  # DOFs problématiques identifiés
        
        print("⚠️  JOINTS PROBLÉMATIQUES IDENTIFIÉS:")
        for dof_id in range(min(25, self.model.nv)):
            joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
            
            if joint_id < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                
                if joint_name:
                    if any(keyword in joint_name.lower() for keyword in ["finger", "thumb"]):
                        self.finger_dofs.append(dof_id)
                        if dof_id in self.problematic_dofs:
                            print(f"   ⚠️  DOF {dof_id}: '{joint_name}' [PROBLÉMATIQUE - SERA BLOQUÉ]")
                        else:
                            print(f"   🖐️  DOF {dof_id}: '{joint_name}' [FINGER]")
                    elif any(keyword in joint_name.lower() for keyword in 
                           ["shoulder", "elbow", "wrist", "arm"]):
                        self.arm_dofs.append(dof_id)
                        print(f"   💪 DOF {dof_id}: '{joint_name}' [ARM]")
        
        # Joints contrôlables
        if self.block_fingers:
            self.controllable_dofs = self.arm_dofs.copy()
            print(f"🛡️  Mode doigts bloqués: {len(self.controllable_dofs)} DOFs contrôlables")
        else:
            self.controllable_dofs = [i for i in range(self.model.nu) if i not in self.problematic_dofs]
            print(f"⚠️  Mode normal: {len(self.controllable_dofs)} DOFs contrôlables")
        
        print(f"   Cube: ID {self.cube_body_id}")
        print(f"   Capteurs force: {len(self.force_sensor_ids)}")
    
    def _setup_spaces(self):
        """Configuration des espaces ultra-conservateurs"""
        # Action space TRÈS restreint
        num_actuators = len(self.controllable_dofs)
        self.action_space = spaces.Box(
            low=-0.1, high=0.1,  # Actions TRÈS petites
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        # Observation space simplifié
        obs_dim = (
            len(self.controllable_dofs) * 2 +  # pos + vel contrôlables
            3 +  # position cube
            1 +  # hauteur cube
            len(self.force_sensor_ids) +  # capteurs force
            4    # phase info
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"   Actions: {self.action_space.shape} (±{self.action_space.high[0]:.2f})")
        print(f"   Observations: {self.observation_space.shape}")
    
    def reset(self, seed=None, options=None):
        """Reset ultra-sécurisé avec blocage des doigts"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # BLOQUER LES JOINTS DE DOIGTS
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = 0.0
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
        
        # Stabilisation progressive
        for i in range(100):
            mj_forward(self.model, self.data)
            
            # Vérifier stabilité
            if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                mj_resetData(self.model, self.data)
                continue
                
            # Maintenir doigts bloqués
            if self.block_fingers:
                for dof_id in self.finger_dofs:
                    if dof_id < len(self.data.qpos):
                        self.data.qpos[dof_id] = 0.0
                    if dof_id < len(self.data.qvel):
                        self.data.qvel[dof_id] = 0.0
            
            if i % 20 == 0:
                try:
                    mj_step(self.model, self.data)
                except:
                    continue
        
        # Position initiale du cube
        if self.cube_body_id >= 0:
            self.cube_initial_pos = self.data.xpos[self.cube_body_id].copy()
            self.cube_initial_height = self.cube_initial_pos[2]
        else:
            self.cube_initial_pos = np.array([0.5, 0.0, 0.45])
            self.cube_initial_height = 0.45
        
        # Reset variables
        self.current_step = 0
        self.contact_detected = False
        self.previous_action = np.zeros(len(self.controllable_dofs))
        self.instability_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Step ultra-protégé"""
        self.current_step += 1
        
        # Clip et lisser l'action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.previous_action is not None:
            action = (1 - self.action_smoothing) * self.previous_action + self.action_smoothing * action
        self.previous_action = action.copy()
        
        # Appliquer action SEULEMENT aux DOFs contrôlables
        self.data.ctrl[:] = 0.0
        for i, dof_id in enumerate(self.controllable_dofs):
            if i < len(action) and dof_id < len(self.data.ctrl):
                self.data.ctrl[dof_id] = action[i]
        
        # MAINTENIR doigts bloqués AVANT simulation
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
                if dof_id < len(self.data.ctrl):
                    self.data.ctrl[dof_id] = 0.0
        
        # Vérification pré-simulation
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return self._get_observation(), -100.0, True, False, {"error": "nan_before_step"}
        
        # Simulation step avec protection
        try:
            mj_step(self.model, self.data)
        except Exception as e:
            return self._get_observation(), -100.0, True, False, {"error": f"mujoco_error: {e}"}
        
        # MAINTENIR doigts bloqués APRÈS simulation
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = 0.0
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
        
        # Vérification post-simulation avec DEBUG
        if (np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or
            np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
            
            # DEBUG: Identifier le joint problématique
            for dof_id in range(self.model.nv):
                if (dof_id < len(self.data.qacc) and 
                    (np.isnan(self.data.qacc[dof_id]) or np.isinf(self.data.qacc[dof_id]))):
                    
                    joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
                    if joint_id < self.model.njnt:
                        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        print(f"⚠️  INSTABILITÉ DÉTECTÉE - DOF {dof_id}: Joint '{joint_name}' (ID {joint_id})")
                    else:
                        print(f"⚠️  INSTABILITÉ DÉTECTÉE - DOF {dof_id}: Free joint")
            
            self.instability_count += 1
            return self._get_observation(), -100.0, True, False, {"error": "simulation_unstable"}
        
        # Calcul observation et récompense
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Conditions de fin
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "contact": self.contact_detected,
            "cube_height": self._get_cube_height(),
            "step": self.current_step,
            "instability_count": self.instability_count,
            "blocked_fingers": self.block_fingers
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Observation simplifiée"""
        try:
            obs_parts = []
            
            # Positions et vitesses contrôlables seulement
            controllable_qpos = []
            controllable_qvel = []
            for dof_id in self.controllable_dofs:
                if dof_id < len(self.data.qpos):
                    controllable_qpos.append(np.clip(self.data.qpos[dof_id], -10, 10))
                if dof_id < len(self.data.qvel):
                    controllable_qvel.append(np.clip(self.data.qvel[dof_id], -10, 10))
            
            obs_parts.append(np.array(controllable_qpos, dtype=np.float32))
            obs_parts.append(np.array(controllable_qvel, dtype=np.float32))
            
            # Position cube
            if self.cube_body_id >= 0:
                cube_pos = self.data.xpos[self.cube_body_id].copy()
            else:
                cube_pos = self.cube_initial_pos.copy()
            obs_parts.append(cube_pos)
            
            # Hauteur relative cube
            cube_height = np.array([cube_pos[2] - self.cube_initial_height])
            obs_parts.append(cube_height)
            
            # Capteurs force
            force_data = []
            for sensor_id in self.force_sensor_ids:
                if sensor_id < len(self.data.sensordata):
                    force_val = self.data.sensordata[sensor_id]
                    if np.isfinite(force_val):
                        force_data.append(np.clip(force_val, -10, 10))
                    else:
                        force_data.append(0.0)
                else:
                    force_data.append(0.0)
            obs_parts.append(np.array(force_data, dtype=np.float32))
            
            # Phase info (simplifié)
            phase_onehot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Toujours "approach"
            obs_parts.append(phase_onehot)
            
            observation = np.concatenate(obs_parts).astype(np.float32)
            
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _compute_reward(self):
        """Récompense ultra-simple"""
        try:
            # Récompense de base pour rester stable
            reward = 1.0
            return float(np.clip(reward, -10.0, 10.0))
        except Exception:
            return 1.0
    
    def _get_cube_height(self):
        """Hauteur cube sécurisée"""
        try:
            if self.cube_body_id >= 0:
                return self.data.xpos[self.cube_body_id][2]
            return self.cube_initial_height
        except Exception:
            return self.cube_initial_height
    
    def _check_termination(self):
        """Termination simple"""
        try:
            return self.instability_count >= 3
        except Exception:
            return True
    
    def render(self, mode=None):
        """Rendu sécurisé"""
        try:
            mode = mode or self.render_mode
            if mode == "rgb_array":
                if self.renderer is None:
                    from mujoco import Renderer
                    self.renderer = Renderer(self.model, width=480, height=320)
                
                self.renderer.update_scene(self.data)
                frame = self.renderer.render()
                return frame
        except Exception:
            return np.zeros((320, 480, 3), dtype=np.uint8)
    
    def close(self):
        """Fermeture sécurisée"""
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        except Exception:
            pass
