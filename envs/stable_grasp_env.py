
#!/usr/bin/env python3
"""
Environnement de saisie stabilisé pour robot G1
Corrige les problèmes de stabilité numérique MuJoCo
Auteur: Assistant IA
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
import time
from pathlib import Path

class StableGraspEnv(gym.Env):
    """
    Environnement stabilisé de saisie du cube pour G1
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 xml_path="results/g1_combined.xml",
                 render_mode=None,
                 max_episode_steps=200,  # Plus court pour éviter instabilité
                 curriculum_level=1):
        super().__init__()
        
        # Configuration
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_level = curriculum_level
        self.current_step = 0
        
        # Charger le modèle MuJoCo avec paramètres de stabilité
        self._load_model()
        
        # Configurer les paramètres de stabilité
        self._configure_stability()
        
        # Identifier les éléments du modèle
        self._identify_model_elements()
        
        # Configurer les espaces d'observation et d'action
        self._setup_spaces()
        
        # Variables d'état
        self.cube_initial_pos = None
        self.cube_initial_height = None
        self.contact_detected = False
        self.grasp_phase = "approach"
        self.phase_start_time = 0.0
        
        # Contrôle des actions
        self.previous_action = None
        self.action_smoothing = 0.1  # Lissage des actions
        
        # Renderer pour les vidéos
        self.renderer = None
        self.viewer = None
        
        print(f"✅ Environnement stabilisé initialisé")
        
    def _load_model(self):
        """Charge le modèle MuJoCo avec gestion d'erreurs"""
        try:
            self.model = MjModel.from_xml_path(self.xml_path)
            self.data = MjData(self.model)
            print(f"✅ Modèle chargé: {self.xml_path}")
            print(f"   Capteurs: {self.model.nsensor}")
            print(f"   Actuateurs: {self.model.nu}")
            print(f"   DOFs: {self.model.nv}")
                
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _configure_stability(self):
        """Configure les paramètres de stabilité MuJoCo"""
        print("🔧 Configuration des paramètres de stabilité...")
        
        # Paramètres de solveur pour la stabilité
        self.model.opt.timestep = 0.005  # Plus grand timestep pour stabilité
        self.model.opt.iterations = 50   # Plus d'itérations du solveur
        self.model.opt.ls_iterations = 20  # Line search iterations
        
        # Paramètres d'intégration
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4  # Runge-Kutta plus stable
        
        # Tolérances plus larges
        self.model.opt.tolerance = 1e-6
        self.model.opt.ls_tolerance = 1e-4
        
        # Amortissement pour stabilité
        self.model.opt.o_solref[0] = 0.02  # Contact softness
        self.model.opt.o_solref[1] = 1.0   # Contact damping
        
        # Limites de vitesse pour éviter les explosions
        if hasattr(self.model, 'jnt_range'):
            # Limiter les vitesses articulaires
            for i in range(self.model.nv):
                if i < len(self.model.dof_damping):
                    self.model.dof_damping[i] = max(0.1, self.model.dof_damping[i])
        
        print("   ✅ Paramètres de stabilité configurés")
    
    def _identify_model_elements(self):
        """Identifie les éléments importants du modèle"""
        # Trouver le cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Identifier les capteurs de force
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
        self.controllable_joints = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                # Joints des bras
                if any(keyword in joint_name.lower() for keyword in 
                       ["shoulder", "elbow", "wrist", "arm", "forearm"]):
                    self.arm_joints.append(i)
                    if i < self.model.nu:  # Vérifier que c'est contrôlable
                        self.controllable_joints.append(i)
                # Joints des doigts
                elif any(keyword in joint_name.lower() for keyword in 
                        ["finger", "thumb", "hand"]):
                    self.finger_joints.append(i)
                    if i < self.model.nu:
                        self.controllable_joints.append(i)
        
        print(f"   Joints bras: {len(self.arm_joints)}")
        print(f"   Joints doigts: {len(self.finger_joints)}")
        print(f"   Joints contrôlables: {len(self.controllable_joints)}")
    
    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        # Action space plus conservateur
        self.action_space = spaces.Box(
            low=-0.5, high=0.5,  # Actions plus petites pour stabilité
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Observation space simplifié
        obs_dim = (
            self.model.nq +   # positions des joints
            self.model.nv  + # vitesses des joints
            3               +# position du cube
            1               +# hauteur du cube
            len(self.force_sensor_ids) +  # capteurs de force
            4                # phase info
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement avec stabilisation"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        
        # Position initiale stable
        if self.model.nq > 0:
            # Positions initiales plus conservatives
            self.data.qpos[:] = 0.0  # Position neutre
            if self.model.nq > 6:  # Si on a plus que les 6 DOF de base
                # Ajouter un très petit bruit pour éviter les singularités
                self.data.qpos[6:] = 0.001 * np.random.randn(self.model.nq - 6)
        
        # Vitesses initiales nulles
        self.data.qvel[:] = 0.0
        
        # Contrôleurs initiaux à zéro
        self.data.ctrl[:] = 0.0
        
        # Simulation forward progressive pour stabilisation
        for i in range(100):  # Plus d'étapes de stabilisation
            mj_forward(self.model, self.data)
            # Vérifier la stabilité
            if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                print(f"⚠️  Instabilité détectée à l'étape {i}, reset")
                mj_resetData(self.model, self.data)
                self.data.qpos[:] = 0.0
                self.data.qvel[:] = 0.0
                self.data.ctrl[:] = 0.0
                continue
            
            # Petite simulation step pour convergence
            if i % 10 == 0:
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
        self.previous_action = np.zeros(self.model.nu)
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécute une étape de simulation avec protection contre l'instabilité"""
        self.current_step = 1
        
        # Clip et lisser l'action pour stabilité
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Lissage des actions
        if self.previous_action is not None:
            action = (1 - self.action_smoothing) * self.previous_action + self.action_smoothing * action
        self.previous_action = action.copy()
        
        # Appliquer l'action
        self.data.ctrl[:] = action
        
        # Vérification avant simulation
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print("⚠️  NaN détecté avant step, terminaison")
            return self._get_observation(), -10.0, True, False, {"error": "nan_detected"}
        
        # Simulation step avec gestion d'erreur
        try:
            mj_step(self.model, self.data)
        except Exception as e:
            print(f"⚠️  Erreur MuJoCo: {e}")
            return self._get_observation(), -10.0, True, False, {"error": str(e)}
        
        # Vérification après simulation
        if (np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or
            np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))):
            print("⚠️  Instabilité détectée, terminaison épisode")
            return self._get_observation(), -10.0, True, False, {"error": "simulation_unstable"}
        
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
        """Calcule l'observation actuelle avec protection"""
        try:
            obs_parts = []
            
            # Positions et vitesses des joints (avec clipping)
            qpos = np.clip(self.data.qpos.copy(), -10, 10)
            qvel = np.clip(self.data.qvel.copy(), -10, 10)
            obs_parts.append(qpos)
            obs_parts.append(qvel)
            
            # Position du cube
            if self.cube_body_id >= 0:
                cube_pos = self.data.xpos[self.cube_body_id].copy()
            else:
                cube_pos = self.cube_initial_pos.copy()
            obs_parts.append(cube_pos)
            
            # Hauteur relative du cube
            cube_height = np.array([cube_pos[2] - self.cube_initial_height])
            obs_parts.append(cube_height)
            
            # Capteurs de force (avec protection)
            force_data = []
            for sensor_id in self.force_sensor_ids:
                if sensor_id < len(self.data.sensordata):
                    force_val = self.data.sensordata[sensor_id]
                    if np.isfinite(force_val):
                        force_data.append(np.clip(force_val, -100, 100))
                    else:
                        force_data.append(0.0)
                else:
                    force_data.append(0.0)
            obs_parts.append(np.array(force_data, dtype=np.float32))
            
            # Phase info
            phase_mapping = {"approach": 0, "contact": 1, "grasp": 2, "lift": 3}
            phase_onehot = np.zeros(4, dtype=np.float32)
            phase_onehot[phase_mapping.get(self.grasp_phase, 0)] = 1.0
            obs_parts.append(phase_onehot)
            
            observation = np.concatenate(obs_parts).astype(np.float32)
            
            # Vérifier que l'observation est valide
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                print("⚠️  Observation invalide détectée")
                observation = np.zeros_like(observation)
            
            return observation
            
        except Exception as e:
            print(f"⚠️  Erreur dans l'observation: {e}")
            # Retourner une observation par défaut
            default_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return default_obs
    
    def _compute_reward(self):
        """Calcule la récompense avec protection"""
        try:
            reward = 0.0
            
            # Récompense de base pour rester stable
            reward = 0.1
            
            # Détection de contact simplifié
            contact_reward = self._compute_contact_reward()
            reward = contact_reward
            
            # Récompense de hauteur
            height_reward = self._compute_height_reward()
            reward = height_reward
            
            # Pénalité plus douce pour les mouvements
            movement_penalty = self._compute_movement_penalty()
            reward = movement_penalty
            
            # Récompense de stabilité
            stability_reward = self._compute_stability_reward()
            reward = stability_reward
            
            # Clamp final
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"⚠️  Erreur dans le calcul de récompense: {e}")
            return 0.1  # Récompense de base
    
    def _compute_contact_reward(self):
        """Récompense pour la détection de contact (simplifiée)"""
        try:
            contact_detected = False
            total_force = 0.0
            
            for sensor_id in self.force_sensor_ids:
                if sensor_id < len(self.data.sensordata):
                    force_magnitude = abs(self.data.sensordata[sensor_id])
                    if np.isfinite(force_magnitude):
                        total_force = force_magnitude
                        if force_magnitude > 0.05:  # Seuil plus bas
                            contact_detected = True
            
            self.contact_detected = contact_detected
            
            if contact_detected:
                return min(0.5, total_force * 0.1)  # Récompense plus modérée
            else:
                return 0.0  # Pas de pénalité
                
        except Exception:
            return 0.0
    
    def _compute_height_reward(self):
        """Récompense pour soulever le cube (simplifiée)"""
        try:
            cube_height = self._get_cube_height()
            height_diff = cube_height - self.cube_initial_height
            
            if height_diff > 0.005:  # Seuil plus bas
                return min(2.0, height_diff * 10)  # Récompense modérée
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compute_movement_penalty(self):
        """Pénalité pour les mouvements excessifs (plus douce)"""
        try:
            action_energy = np.sum(np.square(self.data.ctrl))
            return -0.001 * action_energy  # Pénalité très faible
        except Exception:
            return 0.0
    
    def _compute_stability_reward(self):
        """Récompense pour la stabilité"""
        try:
            # Récompense pour éviter les valeurs extrêmes
            if (np.all(np.isfinite(self.data.qpos)) and 
                np.all(np.isfinite(self.data.qvel)) and
                np.all(np.isfinite(self.data.qacc))):
                return 0.1
            else:
                return -1.0
        except Exception:
            return 0.0
    
    def _get_cube_height(self):
        """Obtient la hauteur actuelle du cube"""
        try:
            if self.cube_body_id >= 0:
                return self.data.xpos[self.cube_body_id][2]
            return self.cube_initial_height
        except Exception:
            return self.cube_initial_height
    
    def _update_phase(self):
        """Met à jour la phase actuelle de la tâche"""
        time_in_phase = self.data.time - self.phase_start_time
        
        if self.grasp_phase == "approach":
            if self.contact_detected or time_in_phase > 5.0:
                self.grasp_phase = "contact"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "contact":
            if time_in_phase > 2.0:
                self.grasp_phase = "grasp"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "grasp":
            if time_in_phase > 3.0:
                self.grasp_phase = "lift"
                self.phase_start_time = self.data.time
    
    def _check_termination(self):
        """Vérifie les conditions de fin d'épisode"""
        try:
            # Succès: cube soulevé
            cube_height = self._get_cube_height()
            if cube_height - self.cube_initial_height > 0.05:
                return True
            
            # Échec: cube au sol
            if cube_height < 0.1:
                return True
            
            return False
            
        except Exception:
            return True  # Terminer en cas d'erreur
    
    def render(self, mode=None):
        """Rendu de l'environnement avec protection"""
        try:
            mode = mode or self.render_mode
            
            if mode == "human":
                if self.viewer is None:
                    from mujoco import viewer as mj_viewer
                    self.viewer = mj_viewer.launch_passive(self.model, self.data)
                self.viewer.sync()
                
            elif mode == "rgb_array":
                if self.renderer is None:
                    from mujoco import Renderer
                    self.renderer = Renderer(self.model, width=640, height=480)
                
                self.renderer.update_scene(self.data)
                frame = self.renderer.render()
                return frame
                
        except Exception as e:
            print(f"⚠️  Erreur de rendu: {e}")
            return None
    
    def close(self):
        """Ferme l'environnement"""
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        except Exception:
            pass
