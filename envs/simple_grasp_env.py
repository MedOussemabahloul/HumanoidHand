
#!/usr/bin/env python3
"""
Environnement simplifié de saisie pour robot G1
Utilise les capteurs de force pour détecter le contact
Auteur: Assistant IA
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
import time
from pathlib import Path

class SimpleGraspEnv(gym.Env):
    """
    Environnement simplifié de saisie du cube pour G1
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 xml_path="/project/results/g1_combined.xml",
                 render_mode=None,
                 max_episode_steps=500,
                 curriculum_level=1):
        super().__init__()
        
        # Configuration
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_level = curriculum_level
        self.current_step = 0
        
        # Charger le modèle MuJoCo
        self._load_model()
        
        # Identifier les éléments du modèle
        self._identify_model_elements()
        
        # Configurer les espaces d'observation et d'action
        self._setup_spaces()
        
        # Variables d'état
        self.cube_initial_pos = None
        self.cube_initial_height = None
        self.contact_detected = False
        self.grasp_phase = "approach"  # approach -> contact -> grasp -> lift
        self.phase_start_time = 0.0
        
        # Renderer pour les vidéos
        self.renderer = None
        self.viewer = None
        
    def _load_model(self):
        """Charge le modèle MuJoCo"""
        try:
            # Changer vers le dossier results pour les chemins relatifs
            original_cwd = Path.cwd()
            results_dir = Path(self.xml_path).parent
            import os
            os.chdir(results_dir)
            
            try:
                self.model = MjModel.from_xml_path(Path(self.xml_path).name)
                self.data = MjData(self.model)
                print(f"✅ Modèle chargé: {self.xml_path}")
                print(f"   Capteurs: {self.model.nsensor}")
                print(f"   Actuateurs: {self.model.nu}")
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _identify_model_elements(self):
        """Identifie les éléments importants du modèle"""
        # Trouver le cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            # Essayer d'autres noms possibles
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Identifier les capteurs de force (pour la détection de contact)
        self.force_sensor_ids = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and ("force" in sensor_name.lower() or "touch" in sensor_name.lower()):
                self.force_sensor_ids.append(i)
        
        print(f"   Cube ID: {self.cube_body_id}")
        print(f"   Capteurs de force: {len(self.force_sensor_ids)}")
        
        # Identifier les joints des bras et mains
        self._identify_arm_joints()
    
    def _identify_arm_joints(self):
        """Identifie les joints des bras et des mains"""
        self.arm_joints = []
        self.finger_joints = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                # Joints des bras (épaules, coudes, poignets)
                if any(keyword in joint_name.lower() for keyword in 
                       ["shoulder", "elbow", "wrist", "arm", "forearm"]):
                    self.arm_joints.append(i)
                # Joints des doigts
                elif any(keyword in joint_name.lower() for keyword in 
                        ["finger", "thumb", "hand"]):
                    self.finger_joints.append(i)
        
        print(f"   Joints bras: {len(self.arm_joints)}")
        print(f"   Joints doigts: {len(self.finger_joints)}")
    
    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        # Action space: contrôle des actuateurs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Observation space: positions, vitesses, position cube, capteurs
        obs_dim = (
            self.model.nq +  # positions des joints
            self.model.nv  + # vitesses des joints
            3             +  # position du cube
            1              + # hauteur du cube
            len(self.force_sensor_ids) +  # capteurs de force
            4                # phase info (one-hot encoding)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        
        # Ajouter un peu de bruit aux positions initiales
        if self.model.nq > 0:
            self.data.qpos[:] = 0.01 * np.random.randn(self.model.nq)
        
        # Position initiale des contrôleurs
        self.data.ctrl[:] = 0.0
        
        # Simulation forward pour stabiliser
        for _ in range(10):
            mj_step(self.model, self.data)
        
        # Enregistrer la position initiale du cube
        if self.cube_body_id >= 0:
            self.cube_initial_pos = self.data.xpos[self.cube_body_id].copy()
            self.cube_initial_height = self.cube_initial_pos[2]
        else:
            self.cube_initial_pos = np.array([0.5, 0.0, 0.45])
            self.cube_initial_height = 0.45
        
        # Reset des variables d'état
        self.current_step = 0
        self.contact_detected = False
        self.grasp_phase = "approach"
        self.phase_start_time = self.data.time
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécute une étape de simulation"""
        self.current_step = 1
        
        # Appliquer l'action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        # Simulation step
        mj_step(self.model, self.data)
        
        # Calculer l'observation et la récompense
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Vérifier les conditions de fin
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Mise à jour de la phase
        self._update_phase()
        
        info = {
            "phase": self.grasp_phase,
            "contact": self.contact_detected,
            "cube_height": self._get_cube_height(),
            "step": self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Calcule l'observation actuelle"""
        obs_parts = []
        
        # Positions et vitesses des joints
        obs_parts.append(self.data.qpos.copy())
        obs_parts.append(self.data.qvel.copy())
        
        # Position du cube
        if self.cube_body_id >= 0:
            cube_pos = self.data.xpos[self.cube_body_id].copy()
        else:
            cube_pos = self.cube_initial_pos.copy()
        obs_parts.append(cube_pos)
        
        # Hauteur relative du cube
        cube_height = np.array([cube_pos[2] - self.cube_initial_height])
        obs_parts.append(cube_height)
        
        # Capteurs de force
        force_data = []
        for sensor_id in self.force_sensor_ids:
            force_data.append(self.data.sensordata[sensor_id])
        obs_parts.append(np.array(force_data, dtype=np.float32))
        
        # Phase info (one-hot)
        phase_mapping = {"approach": 0, "contact": 1, "grasp": 2, "lift": 3}
        phase_onehot = np.zeros(4, dtype=np.float32)
        phase_onehot[phase_mapping.get(self.grasp_phase, 0)] = 1.0
        obs_parts.append(phase_onehot)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _compute_reward(self):
        """Calcule la récompense"""
        reward = 0.0
        
        # Détection de contact
        contact_reward = self._compute_contact_reward()
        reward = contact_reward
        
        # Récompense de hauteur (lift)
        height_reward = self._compute_height_reward()
        reward = height_reward
        
        # Pénalité de mouvement excessif
        movement_penalty = self._compute_movement_penalty()
        reward = movement_penalty
        
        # Récompense de stabilité
        stability_reward = self._compute_stability_reward()
        reward = stability_reward
        
        # Curriculum learning: ajuster les récompenses selon le niveau
        reward *= self._get_curriculum_multiplier()
        
        return float(reward)
    
    def _compute_contact_reward(self):
        """Récompense pour la détection de contact"""
        # Vérifier les capteurs de force
        contact_detected = False
        total_force = 0.0
        
        for sensor_id in self.force_sensor_ids:
            force_magnitude = abs(self.data.sensordata[sensor_id])
            total_force = force_magnitude
            if force_magnitude > 0.1:  # Seuil de détection
                contact_detected = True
        
        self.contact_detected = contact_detected
        
        if contact_detected:
            # Récompense proportionnelle à la force mais limitée
            return min(1.0, total_force * 0.5)
        else:
            return -0.1  # Petite pénalité sans contact
    
    def _compute_height_reward(self):
        """Récompense pour soulever le cube"""
        cube_height = self._get_cube_height()
        height_diff = cube_height - self.cube_initial_height
        
        if height_diff > 0.01:  # Le cube s'élève
            return min(10.0, height_diff * 20)  # Récompense substantielle
        elif height_diff < -0.05:  # Le cube tombe
            return -5.0
        else:
            return 0.0
    
    def _compute_movement_penalty(self):
        """Pénalité pour les mouvements excessifs"""
        # Pénalité basée sur l'énergie des actions
        action_energy = np.sum(np.square(self.data.ctrl))
        return -0.01 * action_energy
    
    def _compute_stability_reward(self):
        """Récompense pour la stabilité du cube"""
        if self.cube_body_id >= 0:
            # Vérifier la stabilité angulaire du cube
            cube_quat = self.data.xquat[self.cube_body_id]
            # Quaternion proche de l'identité = cube stable
            stability = 1.0 - np.linalg.norm(cube_quat - np.array([1,0,0,0]))
            return 0.5 * max(0, stability)
        return 0.0
    
    def _get_curriculum_multiplier(self):
        """Multiplicateur selon le niveau de curriculum"""
        if self.curriculum_level == 1:
            return 1.0  # Niveau de base
        elif self.curriculum_level == 2:
            return 1.2  # Légèrement plus difficile
        elif self.curriculum_level == 3:
            return 1.5  # Plus difficile
        else:
            return 1.0
    
    def _get_cube_height(self):
        """Obtient la hauteur actuelle du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id][2]
        return self.cube_initial_height
    
    def _update_phase(self):
        """Met à jour la phase actuelle de la tâche"""
        time_in_phase = self.data.time - self.phase_start_time
        
        if self.grasp_phase == "approach":
            if self.contact_detected:
                self.grasp_phase = "contact"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "contact":
            if time_in_phase > 1.0:  # 1 seconde de contact
                self.grasp_phase = "grasp"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "grasp":
            if time_in_phase > 2.0:  # 2 secondes pour saisir
                self.grasp_phase = "lift"
                self.phase_start_time = self.data.time
    
    def _check_termination(self):
        """Vérifie les conditions de fin d'épisode"""
        # Succès: cube soulevé suffisamment haut
        cube_height = self._get_cube_height()
        if cube_height - self.cube_initial_height > 0.1:
            return True
        
        # Échec: cube trop bas ou trop éloigné
        if self.cube_body_id >= 0:
            cube_pos = self.data.xpos[self.cube_body_id]
            if cube_pos[2] < 0.1:  # Cube au sol
                return True
            
            # Distance horizontale trop grande
            horizontal_dist = np.linalg.norm(cube_pos[:2] - self.cube_initial_pos[:2])
            if horizontal_dist > 0.5:
                return True
        
        return False
    
    def render(self, mode=None):
        """Rendu de l'environnement"""
        mode = mode or self.render_mode
        
        if mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
        elif mode == "rgb_array":
            if self.renderer is None:
                from mujoco import Renderer
                self.renderer = Renderer(self.model, width=640, height=480)
            
            self.renderer.update_scene(self.data)
            frame = self.renderer.render()
            return frame
            
        else:
            raise ValueError(f"Mode de rendu non supporté: {mode}")
    
    def close(self):
        """Ferme l'environnement"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
