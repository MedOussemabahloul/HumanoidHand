
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
import cv2

class UltraStableGraspEnv(gym.Env):
    """Environnement ultra-stable avec grasping intelligent et capteurs tactiles"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="results/g1_combined.xml", render_mode=None,
                max_episode_steps=50, enable_video_recording=True):
        super().__init__()
        
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.enable_video_recording = enable_video_recording
        
        # Charger le modèle avec correction physique
        self._load_model()
        self._identify_components()
        self._setup_spaces()
        
        # Variables de grasping
        self.cube_initial_pos = None
        self.cube_initial_height = None
        self.grasp_phase = "search"  # search -> approach -> grasp -> lift
        self.contact_detected = False
        self.grasp_strength = 0.0
        self.phase_start_time = 0.0
        
        # Stabilité
        self.previous_action = None
        self.action_smoothing = 0.15
        self.instability_count = 0
        self.max_instabilities = 5
        
        # Enregistrement vidéo
        self.video_frames = []
        self.renderer = None
        
        print(f"✅ Environnement ULTRA-STABLE prêt")
        print(f"   🖐️  Doigts: {len(self.finger_dofs)} DOFs")
        print(f"   💪 Bras: {len(self.arm_dofs)} DOFs")
        print(f"   📱 Capteurs tactiles: {len(self.touch_sensor_ids)}")
        
    def _load_model(self):
        """Charge le modèle avec paramètres ultra-stables"""
        self.model = MjModel.from_xml_path(self.xml_path)
        self.data = MjData(self.model)
        
        # Paramètres de simulation ultra-conservateurs
        self.model.opt.timestep = 0.001  # Très petit pas de temps
        self.model.opt.iterations = 300   # Beaucoup d'itérations
        self.model.opt.ls_iterations = 100
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.tolerance = 1e-10
        self.model.opt.ls_tolerance = 1e-8
        
        # Amortissement global élevé
        self.model.opt.viscosity = 0.1
        
        print("🔧 Modèle chargé avec paramètres ultra-stables")
        
    def _identify_components(self):
        """Identifie tous les composants du modèle"""
        # Cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Classification des DOFs
        self.finger_dofs = []
        self.arm_dofs = []
        self.controllable_dofs = []
        
        finger_keywords = ["finger", "thumb", "index", "middle", "ring"]
        arm_keywords = ["shoulder", "elbow", "wrist", "arm"]
        
        for dof_id in range(min(35, self.model.nv)):
            if dof_id == 0:  # Skip cube DOF
                continue
                
            joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
            
            if joint_id < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                
                if joint_name:
                    joint_lower = joint_name.lower()
                    
                    if any(keyword in joint_lower for keyword in finger_keywords):
                        self.finger_dofs.append(dof_id)
                    elif any(keyword in joint_lower for keyword in arm_keywords):
                        self.arm_dofs.append(dof_id)
                    
                    self.controllable_dofs.append(dof_id)
        
        # Capteurs tactiles
        self.touch_sensor_ids = []
        self.force_sensor_ids = []
        
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name:
                if "touch" in sensor_name.lower() or "contact" in sensor_name.lower():
                    self.touch_sensor_ids.append(i)
                elif "force" in sensor_name.lower():
                    self.force_sensor_ids.append(i)
        
        print(f"   📦 Cube ID: {self.cube_body_id}")
        print(f"   🖐️  Doigts: {self.finger_dofs}")
        print(f"   💪 Bras: {self.arm_dofs}")
        print(f"   📱 Capteurs tactiles: {len(self.touch_sensor_ids)}")
        
    def _setup_spaces(self):
        """Configuration des espaces d'action et d'observation"""
        num_actuators = len(self.controllable_dofs)
        if num_actuators == 0:
            num_actuators = 1
            self.controllable_dofs = [1]
        
        # Actions très limitées pour stabilité
        self.action_space = spaces.Box(
            low=-0.01, high=0.01,  # Actions très petites
            shape=(num_actuators,), 
            dtype=np.float32
        )
        
        # Observations: états joints + cube + capteurs + phase
        obs_dim = (
            len(self.controllable_dofs) * 2 +  # qpos + qvel
            6 +  # cube position + orientation
            max(1, len(self.touch_sensor_ids)) +  # capteurs tactiles
            max(1, len(self.force_sensor_ids)) +  # capteurs force
            4    # phase du grasping
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"   🎯 Actions: {self.action_space.shape} (±{self.action_space.high[0]:.3f})")
        print(f"   👁️  Observations: {self.observation_space.shape}")
        
    def reset(self, seed=None, options=None):
        """Reset avec stabilisation progressive"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # Stabilisation ultra-progressive
        for i in range(1000):  # Plus de steps de stabilisation
            if i % 100 == 0:
                # Vérification de stabilité
                if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                    print(f"⚠️  Instabilité détectée à l'étape {i}, reset partiel")
                    self.data.qvel[:] = 0.0
                    continue
            
            mj_forward(self.model, self.data)
            
            if i % 200 == 0:
                try:
                    mj_step(self.model, self.data)
                except:
                    print(f"⚠️  Step échoué à l'étape {i}")
                    continue
        
        # Position initiale du cube
        if self.cube_body_id >= 0:
            self.cube_initial_pos = self.data.xpos[self.cube_body_id].copy()
            self.cube_initial_height = self.cube_initial_pos[2]
        else:
            self.cube_initial_pos = np.array([0.3, 0.0, 0.05])
            self.cube_initial_height = 0.05
        
        # Reset variables
        self.current_step = 0
        self.grasp_phase = "search"
        self.contact_detected = False
        self.grasp_strength = 0.0
        self.phase_start_time = 0.0
        self.previous_action = np.zeros(len(self.controllable_dofs))
        self.instability_count = 0
        self.video_frames = []
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Step avec gestion intelligente du grasping"""
        self.current_step += 1
        
        # Lissage des actions pour stabilité
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.previous_action is not None:
            action = (1 - self.action_smoothing) * self.previous_action + self.action_smoothing * action
        self.previous_action = action.copy()
        
        # Détection du contact tactile
        self._update_contact_detection()
        
        # Gestion intelligente du grasping
        action = self._intelligent_grasping(action)
        
        # Application des actions
        self.data.ctrl[:] = 0.0
        for i, dof_id in enumerate(self.controllable_dofs):
            if i < len(action) and dof_id < len(self.data.ctrl):
                self.data.ctrl[dof_id] = action[i]
        
        # Vérification pré-step
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return self._get_observation(), -100.0, True, False, {"error": "pre_step_invalid"}
        
        # Step MuJoCo avec gestion d'erreurs
        try:
            mj_step(self.model, self.data)
        except Exception as e:
            print(f"🚨 Erreur MuJoCo step: {e}")
            return self._get_observation(), -100.0, True, False, {"error": f"mujoco_step: {e}"}
        
        # Vérification post-step
        if (np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or
            np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
            
            self.instability_count += 1
            if self.instability_count >= self.max_instabilities:
                return self._get_observation(), -100.0, True, False, {"error": "too_many_instabilities"}
            else:
                # Tentative de récupération
                self.data.qvel[:] = 0.0
                return self._get_observation(), -10.0, False, False, {"error": "instability_recovered"}
        
        # Calculs
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Enregistrement vidéo
        if self.enable_video_recording:
            frame = self.render()
            if frame is not None:
                self.video_frames.append(frame)
        
        info = {
            "contact": self.contact_detected,
            "grasp_phase": self.grasp_phase,
            "cube_height": self._get_cube_height(),
            "step": self.current_step,
            "instability_count": self.instability_count,
            "grasp_strength": self.grasp_strength,
            "video_frames": len(self.video_frames) if self.enable_video_recording else 0
        }
        
        return obs, reward, terminated, truncated, info
        
    def _update_contact_detection(self):
        """Met à jour la détection de contact tactile"""
        contact_sum = 0.0
        
        for sensor_id in self.touch_sensor_ids:
            if sensor_id < len(self.data.sensordata):
                contact_value = self.data.sensordata[sensor_id]
                if np.isfinite(contact_value):
                    contact_sum += abs(contact_value)
        
        # Seuil de contact adaptatif
        self.contact_detected = contact_sum > 0.1
        
    def _intelligent_grasping(self, base_action):
        """Gestion intelligente des phases de grasping"""
        current_time = self.current_step * self.model.opt.timestep
        
        if self.grasp_phase == "search":
            # Phase de recherche: mouvement lent vers le cube
            if current_time - self.phase_start_time > 2.0:  # Après 2 secondes
                cube_pos = self._get_cube_position()
                # Orienter les bras vers le cube (logique simplifiée)
                for i, dof_id in enumerate(self.arm_dofs[:6]):  # Premiers 6 DOFs des bras
                    if i < len(base_action):
                        base_action[i] += 0.001 * np.sign(cube_pos[0] - 0.1)  # Approche progressive
                
                if current_time - self.phase_start_time > 5.0:
                    self.grasp_phase = "approach"
                    self.phase_start_time = current_time
                    print("🤏 Phase: Approche du cube")
        
        elif self.grasp_phase == "approach":
            # Phase d'approche: positionner les mains près du cube
            if self.contact_detected:
                self.grasp_phase = "grasp"
                self.phase_start_time = current_time
                print("✋ Phase: Saisie détectée - fermeture des doigts")
            
        elif self.grasp_phase == "grasp":
            # Phase de saisie: fermer progressivement les doigts
            self.grasp_strength = min(1.0, (current_time - self.phase_start_time) / 3.0)
            
            # Appliquer la fermeture des doigts
            for dof_id in self.finger_dofs:
                dof_index = self.controllable_dofs.index(dof_id) if dof_id in self.controllable_dofs else -1
                if dof_index >= 0 and dof_index < len(base_action):
                    base_action[dof_index] = self.grasp_strength * 0.5  # Fermeture progressive
            
            if current_time - self.phase_start_time > 3.0:
                self.grasp_phase = "lift"
                self.phase_start_time = current_time
                print("⬆️  Phase: Levage du cube")
        
        elif self.grasp_phase == "lift":
            # Phase de levage: maintenir la prise et lever
            # Maintenir la fermeture des doigts
            for dof_id in self.finger_dofs:
                dof_index = self.controllable_dofs.index(dof_id) if dof_id in self.controllable_dofs else -1
                if dof_index >= 0 and dof_index < len(base_action):
                    base_action[dof_index] = 0.8  # Prise ferme
            
            # Mouvement de levage des bras (logique simplifiée)
            for i, dof_id in enumerate(self.arm_dofs):
                if "elbow" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, 
                                            self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id) or "":
                    dof_index = self.controllable_dofs.index(dof_id) if dof_id in self.controllable_dofs else -1
                    if dof_index >= 0 and dof_index < len(base_action):
                        base_action[dof_index] -= 0.002  # Plier les coudes pour lever
        
        return base_action
        
    def _get_cube_position(self):
        """Récupère la position du cube"""
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
            return self.data.xpos[self.cube_body_id].copy()
        return self.cube_initial_pos.copy()
        
    def _get_observation(self):
        """Observation complète avec capteurs tactiles"""
        try:
            obs_parts = []
            
            # États des joints
            qpos = [np.clip(self.data.qpos[i], -5, 5) if i < len(self.data.qpos) else 0.0 
                    for i in self.controllable_dofs]
            qvel = [np.clip(self.data.qvel[i], -5, 5) if i < len(self.data.qvel) else 0.0 
                    for i in self.controllable_dofs]
            
            obs_parts.extend([np.array(qpos, dtype=np.float32), np.array(qvel, dtype=np.float32)])
            
            # Position et orientation du cube
            cube_pos = self._get_cube_position()
            if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xquat):
                cube_quat = self.data.xquat[self.cube_body_id].copy()
            else:
                cube_quat = np.array([1, 0, 0])  # Quaternion identité tronqué
            
            obs_parts.append(cube_pos)
            obs_parts.append(cube_quat)
            
            # Capteurs tactiles
            touch_data = []
            for sensor_id in self.touch_sensor_ids:
                if sensor_id < len(self.data.sensordata):
                    val = self.data.sensordata[sensor_id]
                    touch_data.append(np.clip(val, -10, 10) if np.isfinite(val) else 0.0)
            if not touch_data:
                touch_data = [0.0]
            obs_parts.append(np.array(touch_data, dtype=np.float32))
            
            # Capteurs de force
            force_data = []
            for sensor_id in self.force_sensor_ids:
                if sensor_id < len(self.data.sensordata):
                    val = self.data.sensordata[sensor_id]
                    force_data.append(np.clip(val, -50, 50) if np.isfinite(val) else 0.0)
            if not force_data:
                force_data = [0.0]
            obs_parts.append(np.array(force_data, dtype=np.float32))
            
            # Phase du grasping (one-hot encoding)
            phase_encoding = np.zeros(4, dtype=np.float32)
            phase_map = {"search": 0, "approach": 1, "grasp": 2, "lift": 3}
            phase_encoding[phase_map.get(self.grasp_phase, 0)] = 1.0
            obs_parts.append(phase_encoding)
            
            observation = np.concatenate(obs_parts).astype(np.float32)
            
            # Vérification de validité
            if (np.any(np.isnan(observation)) or np.any(np.isinf(observation)) or
                len(observation) != self.observation_space.shape[0]):
                observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            return observation
            
        except Exception:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
    def _compute_reward(self):
        """Récompense basée sur le succès du grasping"""
        try:
            reward = 1.0  # Base pour survivre
            
            # Bonus pour stabilité
            if self.instability_count == 0:
                reward += 2.0
            
            # Bonus pour progression des phases
            phase_rewards = {"search": 0.5, "approach": 1.0, "grasp": 2.0, "lift": 5.0}
            reward += phase_rewards.get(self.grasp_phase, 0.0)
            
            # Bonus pour contact tactile
            if self.contact_detected:
                reward += 3.0
            
            # Bonus pour levage du cube
            cube_height = self._get_cube_height()
            height_bonus = max(0, (cube_height - self.cube_initial_height) * 100)
            reward += height_bonus
            
            # Bonus pour utilisation coordonnée des doigts
            if self.grasp_phase in ["grasp", "lift"]:
                finger_activity = sum(abs(self.data.qvel[dof_id]) for dof_id in self.finger_dofs 
                                    if dof_id < len(self.data.qvel))
                if finger_activity > 0.01:
                    reward += 1.0
            
            return float(np.clip(reward, -100.0, 100.0))
            
        except Exception:
            return 1.0
            
    def _get_cube_height(self):
        """Hauteur du cube sécurisée"""
        try:
            cube_pos = self._get_cube_position()
            return cube_pos[2] if np.isfinite(cube_pos[2]) else self.cube_initial_height
        except Exception:
            return self.cube_initial_height
            
    def _check_termination(self):
        """Termination seulement si échec critique"""
        return self.instability_count >= self.max_instabilities
        
    def render(self, mode=None):
        """Rendu avec support vidéo"""
        try:
            if mode == "rgb_array" or self.render_mode == "rgb_array" or self.enable_video_recording:
                if self.renderer is None:
                    from mujoco import Renderer
                    self.renderer = Renderer(self.model, width=640, height=480)
                self.renderer.update_scene(self.data)
                return self.renderer.render()
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
    def save_video(self, filename="training_video.mp4", fps=30):
        """Sauvegarde la vidéo d'entraînement"""
        if not self.video_frames or not self.enable_video_recording:
            print("⚠️  Aucune frame vidéo à sauvegarder")
            return False
            
        try:
            height, width = self.video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            for frame in self.video_frames:
                # Convertir RGB en BGR pour OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            print(f"✅ Vidéo sauvegardée: {filename} ({len(self.video_frames)} frames)")
            return True
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde vidéo: {e}")
            return False
            
    def close(self):
        """Fermeture avec sauvegarde vidéo"""
        try:
            if self.enable_video_recording and self.video_frames:
                self.save_video("final_training_video.mp4")
            
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        except Exception:
            pass
