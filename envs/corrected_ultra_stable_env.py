import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward

class CorrectedUltraStableGraspEnv(gym.Env):
  """Environnement avec correction physique des doigts (sans blocage)"""
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self, xml_path="results/g1_combined.xml", render_mode=None,
             max_episode_steps=30, fix_physics=True):
      super().__init__()
      
      self.xml_path = xml_path
      self.render_mode = render_mode
      self.max_episode_steps = max_episode_steps
      self.current_step = 0
      self.fix_physics = fix_physics
      
      # Charger et corriger le modèle
      self._load_and_fix_model()
      self._identify_joints()
      self._setup_spaces()
      
      # Variables
      self.cube_initial_pos = None
      self.cube_initial_height = None
      self.contact_detected = False
      self.previous_action = None
      self.action_smoothing = 0.1
      self.instability_count = 0
      self.renderer = None
      
      print(f"✅ Environnement PHYSIQUE CORRIGÉ prêt")
      print(f"   🖐️  Doigts: {len(self.finger_dofs)} DOFs (ACTIFS)")
      print(f"   💪 Bras: {len(self.arm_dofs)} DOFs")
      
  def _load_and_fix_model(self):
      """Charge et CORRIGE les paramètres physiques"""
      self.model = MjModel.from_xml_path(self.xml_path)
      self.data = MjData(self.model)
      
      if self.fix_physics:
          print("🔧 CORRECTION PHYSIQUE DES DOIGTS...")
          
          # 1. SOLVER plus robuste
          self.model.opt.timestep = 0.005  # Plus grand pour stabilité
          self.model.opt.iterations = 100   # Plus d'itérations
          self.model.opt.ls_iterations = 50
          self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4  # Plus stable
          self.model.opt.tolerance = 1e-8
          self.model.opt.ls_tolerance = 1e-6
          
          # 2. AMORTISSEMENT ADAPTATIF pour doigts
          finger_keywords = ["finger", "thumb", "index", "middle", "ring"]
          for dof_id in range(self.model.nv):
              if dof_id < self.model.njnt:
                  joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, dof_id)
                  if joint_name and any(kw in joint_name.lower() for kw in finger_keywords):
                      # Amortissement ÉLEVÉ pour doigts
                      if dof_id < len(self.model.dof_damping):
                          self.model.dof_damping[dof_id] = 5.0  # 500x plus élevé
                      print(f"   🖐️  DOF {dof_id} ({joint_name}): damping = 5.0")
                  else:
                      # Amortissement normal pour bras
                      if dof_id < len(self.model.dof_damping):
                          self.model.dof_damping[dof_id] = max(0.5, self.model.dof_damping[dof_id])
          
          # 3. GAINS D'ACTUATEURS RÉDUITS pour doigts
          for act_id in range(self.model.nu):
              actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
              if actuator_name and any(kw in actuator_name.lower() for kw in finger_keywords):
                  # Gains RÉDUITS pour doigts
                  if hasattr(self.model, 'actuator_gainprm'):
                      self.model.actuator_gainprm[act_id][0] = 20.0  # kp réduit
                      self.model.actuator_gainprm[act_id][1] = 2.0   # kv réduit
                      print(f"   🎛️  Actuateur {act_id} ({actuator_name}): gains réduits")
          
          # 4. LIMITES DE FORCE
          for act_id in range(self.model.nu):
              if hasattr(self.model, 'actuator_forcerange'):
                  self.model.actuator_forcerange[act_id][0] = -10.0  # Force max réduite
                  self.model.actuator_forcerange[act_id][1] = 10.0
      
      print(f"✅ Modèle physiquement corrigé: {self.model.nv} DOFs, {self.model.nu} actuateurs")
  
  def _identify_joints(self):
      """Identification normale (sans blocage)"""
      print("🔧 IDENTIFICATION DES JOINTS...")
      
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
      
      # Classification des joints
      self.finger_dofs = []
      self.arm_dofs = []
      
      finger_keywords = ["finger", "thumb", "index", "middle", "ring", "pinky"]
      
      for dof_id in range(min(31, self.model.nv)):
          joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
          
          if joint_id < self.model.njnt:
              joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
              
              if joint_name:
                  joint_lower = joint_name.lower()
                  
                  if any(keyword in joint_lower for keyword in finger_keywords):
                      self.finger_dofs.append(dof_id)
                      print(f"DOF {dof_id:2d}: {joint_name:25s} [🖐️  FINGER ACTIF]")
                  elif any(kw in joint_lower for kw in ["shoulder", "elbow", "wrist", "arm"]):
                      self.arm_dofs.append(dof_id)
                      print(f"DOF {dof_id:2d}: {joint_name:25s} [💪 ARM]")
                  else:
                      print(f"DOF {dof_id:2d}: {joint_name:25s} [🤖 OTHER]")
      
      # TOUS les DOFs sont contrôlables (sauf cube)
      self.controllable_dofs = [i for i in range(1, min(31, self.model.nv))]  # Exclure cube (DOF 0)
      
      print(f"\n🎯 CONFIGURATION PHYSIQUE CORRIGÉE:")
      print(f"   🖐️  Doigts ACTIFS: {self.finger_dofs}")
      print(f"   💪 Bras ACTIFS: {self.arm_dofs}")
      print(f"   🎯 Contrôlables: {len(self.controllable_dofs)} DOFs")
      print(f"   📦 Cube ID: {self.cube_body_id}")
      print(f"   📊 Capteurs: {len(self.force_sensor_ids)}")
  
  def _setup_spaces(self):
      """Configuration des espaces"""
      num_actuators = len(self.controllable_dofs)
      if num_actuators == 0:
          num_actuators = 1
          self.controllable_dofs = [1]
          
      # Actions plus conservatrices pour stabilité
      self.action_space = spaces.Box(
          low=-0.02, high=0.02,  # Actions très petites
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
      """Reset avec stabilisation physique"""
      super().reset(seed=seed)
      
      # Reset MuJoCo
      mj_resetData(self.model, self.data)
      self.data.qpos[:] = 0.0
      self.data.qvel[:] = 0.0
      self.data.ctrl[:] = 0.0
      
      # Stabilisation progressive
      for i in range(500):  # Plus de steps de stabilisation
          mj_forward(self.model, self.data)
          
          # Vérification de stabilité
          if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
              print(f"⚠️  Instabilité détectée à l'étape {i}")
              # Reset partiel
              self.data.qvel[:] = 0.0
              continue
          
          if i % 100 == 0:
              try:
                  mj_step(self.model, self.data)
              except:
                  print(f"⚠️  Step échoué à l'étape {i}")
                  continue
      
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
      """Step avec contrôle physique amélioré"""
      self.current_step += 1
      
      # Actions ultra-conservative avec lissage
      action = np.clip(action, self.action_space.low, self.action_space.high)
      if self.previous_action is not None:
          action = (1 - self.action_smoothing) * self.previous_action + self.action_smoothing * action
      self.previous_action = action.copy()
      
      # Reset complet des contrôles
      self.data.ctrl[:] = 0.0
      
      # Appliquer actions à TOUS les DOFs contrôlables (y compris doigts)
      for i, dof_id in enumerate(self.controllable_dofs):
          if i < len(action) and dof_id < len(self.data.ctrl):
              self.data.ctrl[dof_id] = action[i]
      
      # Vérification pré-step
      if (np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
          return self._get_observation(), -50.0, True, False, {"error": "pre_step_invalid"}
      
      # Step MuJoCo avec gestion d'erreurs
      try:
          mj_step(self.model, self.data)
      except Exception as e:
          print(f"🚨 Erreur MuJoCo step: {e}")
          return self._get_observation(), -50.0, True, False, {"error": f"mujoco_step: {e}"}
      
      # Vérification post-step
      if (np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or
          np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
          
          print("🚨 INSTABILITÉ PHYSIQUE DÉTECTÉE:")
          for dof_id in range(min(31, self.model.nv)):
              if (dof_id < len(self.data.qacc) and 
                  (np.isnan(self.data.qacc[dof_id]) or np.isinf(self.data.qacc[dof_id]))):
                  
                  joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
                  if joint_id < self.model.njnt:
                      joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                      finger_status = "FINGER" if dof_id in self.finger_dofs else "ARM"
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
          "active_fingers": len(self.finger_dofs),
          "controllable_dofs": len(self.controllable_dofs)
      }
      
      return obs, reward, terminated, truncated, info
  
  def _get_observation(self):
      """Observation sécurisée"""
      try:
          obs_parts = []
          
          # États des joints (TOUS)
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
      """Récompense avec bonus pour utilisation des doigts"""
      try:
          reward = 2.0  # Base pour survivre
          
          if self.instability_count == 0:
              reward += 1.0  # Bonus stabilité
          
          # Bonus pour mouvement coordonné des doigts
          finger_activity = 0.0
          for dof_id in self.finger_dofs:
              if dof_id < len(self.data.qvel):
                  finger_activity += abs(self.data.qvel[dof_id])
          
          if finger_activity > 0.01:  # Si les doigts bougent
              reward += 0.5
          
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
      """Termination seulement si vraiment instable"""
      return self.instability_count >= 3  # Plus tolérant
  
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
