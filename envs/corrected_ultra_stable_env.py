
#!/usr/bin/env python3
"""
ENVIRONNEMENT CORRIGÉ - Identification exacte des doigts
+Corrige définitivement l'identification des DOF 15-30
+"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward

class CorrectedUltraStableGraspEnv(gym.Env):
    """Environnement avec identification CORRIGÉE de TOUS les doigts"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="results/g1_combined.xml", render_mode=None,
               max_episode_steps=30, curriculum_level=1, block_fingers=True):
      super().__init__()
      
      self.xml_path = xml_path
      self.render_mode = render_mode
      self.max_episode_steps = max_episode_steps
      self.current_step = 0
      self.block_fingers = block_fingers
      
      # Charger modèle
      self._load_and_configure_model()
      self._identify_joints_corrected()
      self._setup_spaces()
      
      # Variables
      self.cube_initial_pos = None
      self.cube_initial_height = None
      self.contact_detected = False
      self.previous_action = None
      self.action_smoothing = 0.03
      self.instability_count = 0
      self.renderer = None
      
      print(f"✅ Environnement CORRIGÉ prêt")
      print(f"   🖐️  Doigts identifiés: {len(self.finger_dofs)} DOFs")
      print(f"   💪 Bras contrôlables: {len(self.arm_dofs)} DOFs")
      
    def _load_and_configure_model(self):
        """Charge et configure avec stabilité maximale"""
        self.model = MjModel.from_xml_path(self.xml_path)
        self.data = MjData(self.model)
        
        # CONFIGURATION ULTRA-STABLE
        self.model.opt.timestep = 0.01
        self.model.opt.iterations = 150
        self.model.opt.ls_iterations = 75
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.model.opt.tolerance = 1e-6
        self.model.opt.ls_tolerance = 1e-4
        
        # Amortissement ultra-élevé
        for i in range(self.model.nv):
            if i < len(self.model.dof_damping):
                self.model.dof_damping[i] = max(1.0, self.model.dof_damping[i] * 20)
        
        print(f"✅ Modèle configuré: {self.model.nv} DOFs, {self.model.nu} actuateurs")
    
    def _identify_joints_corrected(self):
        """IDENTIFICATION CORRIGÉE - détecte TOUS les doigts"""
        print("🔧 IDENTIFICATION CORRIGÉE DES JOINTS...")
        
        # Cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Capteurs force
        self.force_sensor_ids = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and ("force" in sensor_name.lower() or "touch" in sensor_name.lower()):
                self.force_sensor_ids.append(i)
        
        # IDENTIFICATION CORRIGÉE COMPLÈTE
        self.finger_dofs = []
        self.arm_dofs = []
        
        # Mots-clés COMPLETS pour doigts
        finger_keywords = [
            "finger", "thumb", "index", "middle", "ring", "pinky", "pinkie"
        ]
        
        # LISTE EXACTE des DOFs problématiques de votre debug
        self.problematic_dofs = [15, 16, 17, 18, 19, 20, 21, 22, 29, 30]
        
        print("🔧 MAPPING CORRIGÉ:")
        print("-" * 50)
        
        # Identifier tous les joints
        for dof_id in range(min(31, self.model.nv)):
            joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
            
            if joint_id < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                
                if joint_name:
                    joint_lower = joint_name.lower()
                    
                    # CORRECTION: Détecter TOUS les types de doigts
                    is_finger = any(keyword in joint_lower for keyword in finger_keywords)
                    
                    # CORRECTION SPÉCIFIQUE: index, middle, ring
                    if not is_finger:
                        for finger_type in ["index", "middle", "ring"]:
                            if finger_type in joint_lower:
                                is_finger = True
                                break
                    
                    if is_finger:
                        self.finger_dofs.append(dof_id)
                        status = "⚠️  PROBLÉMATIQUE" if dof_id in self.problematic_dofs else ""
                        print(f"DOF {dof_id:2d}: {joint_name:25s} [🖐️  FINGER {status}]")
                    
                    elif any(kw in joint_lower for kw in ["shoulder", "elbow", "wrist", "arm"]):
                        self.arm_dofs.append(dof_id)
                        print(f"DOF {dof_id:2d}: {joint_name:25s} [💪 ARM]")
                    
                    else:
                        print(f"DOF {dof_id:2d}: {joint_name:25s} [🤖 OTHER]")
        
        # VÉRIFICATION et CORRECTION FORCÉE
        missing_fingers = set(self.problematic_dofs) - set(self.finger_dofs)
        if missing_fingers:
            print(f"\n🚨 CORRECTION FORCÉE - DOFs manqués: {list(missing_fingers)}")
            for dof_id in missing_fingers:
                if dof_id < self.model.nv:
                    joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
                    if joint_id < self.model.njnt:
                        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        print(f"   🔧 Ajout forcé DOF {dof_id}: '{joint_name}' → FINGER")
                        self.finger_dofs.append(dof_id)
        
        # Trier et finaliser
        self.finger_dofs = sorted(list(set(self.finger_dofs)))
        self.arm_dofs = sorted(list(set(self.arm_dofs)))
        
        # Configuration finale
        if self.block_fingers:
            self.controllable_dofs = self.arm_dofs.copy()
            print(f"\n🛡️  CONFIGURATION FINALE:")
            print(f"   🖐️  Doigts BLOQUÉS: {self.finger_dofs}")
            print(f"   💪 Bras ACTIFS: {self.arm_dofs}")
            print(f"   🎯 Contrôlables: {len(self.controllable_dofs)} DOFs")
        else:
            self.controllable_dofs = [i for i in range(self.model.nu) if i not in self.problematic_dofs]
        
        print(f"   📦 Cube ID: {self.cube_body_id}")
        print(f"   📊 Capteurs: {len(self.force_sensor_ids)}")
    
    def _setup_spaces(self):
        """Configuration ultra-conservative"""
        num_actuators = len(self.controllable_dofs)
        if num_actuators == 0:
            num_actuators = 1
            self.controllable_dofs = [1]
            
        self.action_space = spaces.Box(
            low=-0.03, high=0.03,  # Actions TRÈS petites
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        obs_dim = (
            len(self.controllable_dofs) * 2 + 3 + 1 + 
            max(1, len(self.force_sensor_ids)) + 4
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"   🎯 Actions: {self.action_space.shape} (±{self.action_space.high[0]:.3f})")
        print(f"   👁️  Observations: {self.observation_space.shape}")
    
    def reset(self, seed=None, options=None):
        """Reset ultra-sécurisé avec blocage TOTAL des doigts"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # BLOCAGE TOTAL des doigts
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = 0.0
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
                if dof_id < len(self.model.dof_damping):
                    self.model.dof_damping[dof_id] = 50.0  # Amortissement extrême
        
        # Stabilisation renforcée
        for attempt in range(3):
            stable = True
            for i in range(200):
                mj_forward(self.model, self.data)
                
                if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                    stable = False
                    break
                    
                # Maintenir blocage des doigts
                if self.block_fingers:
                    for dof_id in self.finger_dofs:
                        if dof_id < len(self.data.qpos):
                            self.data.qpos[dof_id] = 0.0
                        if dof_id < len(self.data.qvel):
                            self.data.qvel[dof_id] = 0.0
                
                if i % 40 == 0:
                    try:
                        mj_step(self.model, self.data)
                    except:
                        stable = False
                        break
            
            if stable:
                break
            else:
                print(f"⚠️  Reset instable, tentative {attempt+1}/3")
                mj_resetData(self.model, self.data)
        
        # Position cube
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
        """Step ultra-protégé avec blocage renforcé"""
        self.current_step += 1
        
        # Actions ultra-conservative
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.previous_action is not None:
            action = (1 - self.action_smoothing) * self.previous_action + self.action_smoothing * action
        self.previous_action = action.copy()
        
        # Reset complet des contrôles
        self.data.ctrl[:] = 0.0
        
        # Appliquer SEULEMENT aux bras
        for i, dof_id in enumerate(self.controllable_dofs):
            if i < len(action) and dof_id < len(self.data.ctrl):
                self.data.ctrl[dof_id] = action[i]
        
        # FORCER blocage doigts AVANT
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = 0.0
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
                if dof_id < len(self.data.ctrl):
                    self.data.ctrl[dof_id] = 0.0
        
        # Vérification pré-step
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
            return self._get_observation(), -50.0, True, False, {"error": "pre_step_invalid"}
        
        # Step MuJoCo
        try:
            mj_step(self.model, self.data)
        except Exception as e:
            return self._get_observation(), -50.0, True, False, {"error": f"mujoco_step: {e}"}
        
        # FORCER blocage doigts APRÈS
        if self.block_fingers:
            for dof_id in self.finger_dofs:
                if dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = 0.0
                if dof_id < len(self.data.qvel):
                    self.data.qvel[dof_id] = 0.0
        
        # Vérification post-step avec DEBUG
        if (np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or
            np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
            
            print("🚨 INSTABILITÉ DÉTECTÉE:")
            for dof_id in range(min(31, self.model.nv)):
                if (dof_id < len(self.data.qacc) and 
                    (np.isnan(self.data.qacc[dof_id]) or np.isinf(self.data.qacc[dof_id]))):
                    
                    joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
                    if joint_id < self.model.njnt:
                        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        finger_status = "FINGER(BLOQUÉ)" if dof_id in self.finger_dofs else "NON-FINGER"
                        print(f"   🚨 DOF {dof_id}: '{joint_name}' [{finger_status}]")
            
            self.instability_count += 1
            return self._get_observation(), -50.0, True, False, {"error": "post_step_unstable"}
        
        # Calculs
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "contact": self.contact_detected,
            "cube_height": self._get_cube_height(),
            "step": self.current_step,
            "instability_count": self.instability_count,
            "blocked_fingers": len(self.finger_dofs),
            "controllable_dofs": len(self.controllable_dofs)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Observation ultra-sécurisée"""
        try:
            obs_parts = []
            
            # Bras seulement
            qpos = [np.clip(self.data.qpos[i], -5, 5) if i < len(self.data.qpos) else 0.0 
                    for i in self.controllable_dofs]
            qvel = [np.clip(self.data.qvel[i], -5, 5) if i < len(self.data.qvel) else 0.0 
                    for i in self.controllable_dofs]
            
            if not qpos:
                qpos = [0.0]
            if not qvel:
                qvel = [0.0]
                
            obs_parts.extend([np.array(qpos, dtype=np.float32), np.array(qvel, dtype=np.float32)])
            
            # Cube
            if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
                cube_pos = self.data.xpos[self.cube_body_id].copy()
                if not np.all(np.isfinite(cube_pos)):
                    cube_pos = self.cube_initial_pos.copy()
            else:
                cube_pos = self.cube_initial_pos.copy()
            
            obs_parts.append(cube_pos)
            obs_parts.append(np.array([cube_pos[2] - self.cube_initial_height]))
            
            # Force
            force_data = []
            if self.force_sensor_ids:
                for sid in self.force_sensor_ids:
                    if sid < len(self.data.sensordata):
                        val = self.data.sensordata[sid]
                        force_data.append(np.clip(val, -3, 3) if np.isfinite(val) else 0.0)
                    else:
                        force_data.append(0.0)
            else:
                force_data = [0.0]
            obs_parts.append(np.array(force_data, dtype=np.float32))
            
            # Phase
            obs_parts.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            
            observation = np.concatenate(obs_parts).astype(np.float32)
            
            if (np.any(np.isnan(observation)) or np.any(np.isinf(observation)) or
                len(observation) != self.observation_space.shape[0]):
                observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            return observation
            
        except Exception:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _compute_reward(self):
        """Récompense de base pour stabilité"""
        try:
            reward = 2.0  # Base pour survivre
            if self.instability_count == 0:
                reward += 1.0  # Bonus stabilité
            return float(np.clip(reward, -10.0, 10.0))
        except Exception:
            return 2.0
    
    def _get_cube_height(self):
        """Hauteur cube sécurisée"""
        try:
            if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
                h = self.data.xpos[self.cube_body_id][2]
                return h if np.isfinite(h) else self.cube_initial_height
            return self.cube_initial_height
        except Exception:
            return self.cube_initial_height
    
    def _check_termination(self):
        """Termination dès la première instabilité"""
        return self.instability_count >= 1
    
    def render(self, mode=None):
        """Rendu sécurisé"""
        try:
            if mode == "rgb_array" or self.render_mode == "rgb_array":
                if self.renderer is None:
                    from mujoco import Renderer
                    self.renderer = Renderer(self.model, width=320, height=240)
                self.renderer.update_scene(self.data)
                return self.renderer.render()
        except Exception:
            return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def close(self):
        """Fermeture sécurisée"""
        try:
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        except Exception:
            pass
