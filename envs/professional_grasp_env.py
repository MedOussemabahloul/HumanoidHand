
#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT DE GRASPING PROFESSIONNEL
===========================================

Fonctionnalités:
✅ Stabilité des bras avec damping adaptatif
✅ Contact palm-cube professionnel
✅ Grasping contrôlé en phases
✅ Collisions physiques réelles
✅ Détection de contact précise
✅ Récupération d'erreurs automatique
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional
import tempfile
import warnings
warnings.filterwarnings("ignore")

class ProfessionalGraspEnv(gym.Env):
    """
    🏆 Environnement de Grasping Professionnel
    
    Phases de grasping:
    1. STABILIZE - Stabiliser les bras
    2. APPROACH - Approcher le cube
    3. CONTACT - Établir contact palm-cube
    4. GRASP - Fermer les doigts
    5. LIFT - Soulever le cube
    6. HOLD - Maintenir la prise
    """
    
    def __init__(self, model_path: str = None, render_mode: str = None, fix_physics: bool = True):
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.fix_physics = fix_physics
        self.model_path_str = model_path or "/home/oussema/Documents/project/results/g1_combined.xml"
        
        # Phases de grasping
        self.PHASES = {
            'STABILIZE': 0,
            'APPROACH': 1, 
            'CONTACT': 2,
            'GRASP': 3,
            'LIFT': 4,
            'HOLD': 5
        }
        
        # État
        self.current_phase = self.PHASES['STABILIZE']
        self.phase_timer = 0
        self.phase_durations = {
            'STABILIZE': 50,
            'APPROACH': 100,
            'CONTACT': 30,
            'GRASP': 50,
            'LIFT': 50,
            'HOLD': 70
        }
        
        # Initialisation du modèle
        self._setup_model()
        self._identify_components()
        self._setup_spaces()
        
        # Métriques
        self.episode_metrics = {
            'phase_completions': {phase: 0 for phase in self.PHASES.keys()},
            'contact_detections': 0,
            'successful_grasps': 0,
            'arm_stability_score': 0.0,
            'palm_contact_score': 0.0,
            'cube_height_max': 0.0
        }
        
        # Enregistrement vidéo
        self.video_frames = []
        self.record_video = False
        
        print("🏆 ProfessionalGraspEnv initialisé avec succès!")
        
    def _setup_model(self):
        """Configuration du modèle avec physique optimisée"""
        try:
            # Créer un fichier temporaire avec physique corrigée
            with open(self.model_path_str, 'r') as f:
                xml_content = f.read()
            
            if self.fix_physics:
                # Appliquer corrections physiques ultra-stables
                xml_content = self._apply_physics_fixes(xml_content)
            
            # Sauvegarder temporairement
            self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.temp_file.write(xml_content)
            self.temp_file.close()
            
            # Charger le modèle
            self.model = mujoco.MjModel.from_xml_path(self.temp_file.name)
            self.data = mujoco.MjData(self.model)
            
            # Renderer pour vidéo
            if self.render_mode == 'rgb_array':
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                
            print(f"✅ Modèle chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _apply_physics_fixes(self, xml_content: str) -> str:
        """Applique les corrections physiques pour ultra-stabilité"""
        
        # Corrections pour les options globales
        fixes = [
            ('timestep="0.002"', 'timestep="0.001"'),
            ('iterations="200"', 'iterations="300"'),
            ('tolerance="1e-8"', 'tolerance="1e-10"'),
        ]
        
        for old, new in fixes:
            if old in xml_content:
                xml_content = xml_content.replace(old, new)
        
        # Corriger les chemins relatifs
        xml_content = xml_content.replace('../assets/hands/', '/home/oussema/Documents/project/assets/hands/')
        
        return xml_content
    
    def _identify_components(self):
        """Identifie tous les composants du robot"""
        
        # DOFs des doigts (15-30)
        self.finger_dofs = list(range(15, 31))
        
        # DOFs des bras (1-14) 
        self.arm_dofs = list(range(1, 15))
        
        # DOFs du corps (0)
        self.body_dofs = [0]
        
        # Capteurs tactiles
        self.touch_sensors = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and 'tip_sensor' in sensor_name:
                self.touch_sensors.append(i)
        
        print(f"🔍 Composants identifiés:")
        print(f"  - Doigts: {len(self.finger_dofs)} DOFs")
        print(f"  - Bras: {len(self.arm_dofs)} DOFs") 
        print(f"  - Capteurs tactiles: {len(self.touch_sensors)}")
    
    def _setup_spaces(self):
        """Configuration des espaces d'observation et d'action"""
        
        # Espace d'action: contrôle des bras et doigts
        action_dim = len(self.arm_dofs) + len(self.finger_dofs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # Espace d'observation
        obs_dim = (
            len(self.arm_dofs) + len(self.finger_dofs) +
            len(self.arm_dofs) + len(self.finger_dofs) +
            7 +
            len(self.touch_sensors) +
            6
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        print(f"📊 Espaces configurés:")
        print(f"  - Actions: {self.action_space.shape}")
        print(f"  - Observations: {self.observation_space.shape}")
    
    def reset(self, seed=None, options=None):
        """Reset de l'environnement"""
        super().reset(seed=seed)
        
        # Reset du modèle
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale stable
        self._set_initial_positions()
        
        # Reset des métriques
        self.current_phase = self.PHASES['STABILIZE']
        self.phase_timer = 0
        self.episode_metrics = {
            'phase_completions': {phase: 0 for phase in self.PHASES.keys()},
            'contact_detections': 0,
            'successful_grasps': 0,
            'arm_stability_score': 0.0,
            'palm_contact_score': 0.0,
            'cube_height_max': 0.0
        }
        
        # Reset vidéo
        self.video_frames = []
        
        # Simulation initiale
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _set_initial_positions(self):
        """Configure les positions initiales stables"""
        
        # Position des bras - configuration stable
        arm_positions = [
            0.0, 0.2, 0.0, -0.5, 0.0, 0.0, 0.0,
            0.0, -0.2, 0.0, -0.5, 0.0, 0.0, 0.0
        ]
        
        # Position des doigts - ouverts
        finger_positions = [0.0] * len(self.finger_dofs)
        
        # Appliquer les positions
        for i, dof in enumerate(self.arm_dofs):
            if i < len(arm_positions):
                self.data.qpos[dof] = arm_positions[i]
        
        for i, dof in enumerate(self.finger_dofs):
            if i < len(finger_positions):
                self.data.qpos[dof] = finger_positions[i]
        
        # Position du cube (fixe sur la table)
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        if cube_body_id >= 0:
            cube_qpos_start = self.model.body_jntadr[cube_body_id]
            if cube_qpos_start >= 0:
                # Position: sur la table
                self.data.qpos[cube_qpos_start:cube_qpos_start+3] = [0.5, 0.0, 0.05]
                # Orientation: stable
                self.data.qpos[cube_qpos_start+3:cube_qpos_start+7] = [1, 0, 0, 0]
    
    def step(self, action):
        """Exécute une action dans l'environnement"""
        
        # Appliquer l'action avec stabilité
        self._apply_stable_action(action)
        
        # Simulation
        mujoco.mj_step(self.model, self.data)
        
        # Vérifier la stabilité
        if self._check_instability():
            return self._handle_instability()
        
        # Gestion des phases
        self._update_phase()
        
        # Calcul de la récompense
        reward = self._compute_reward()
        
        # Vérification de fin d'épisode
        terminated, truncated = self._check_termination()
        
        # Observation et info
        observation = self._get_observation()
        info = self._get_info()
        
        # Enregistrement vidéo
        if self.record_video:
            self._record_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_stable_action(self, action):
        """Applique l'action avec contrôle de stabilité"""
        
        # Normaliser l'action
        action = np.clip(action, -1.0, 1.0)
        
        # Séparation bras/doigts
        arm_action = action[:len(self.arm_dofs)]
        finger_action = action[len(self.arm_dofs):]
        
        # Contrôle adaptatif selon la phase
        phase_name = list(self.PHASES.keys())[self.current_phase]
        
        if phase_name == 'STABILIZE':
            arm_action *= 0.1
            finger_action *= 0.05
        elif phase_name == 'APPROACH':
            arm_action = self._guided_approach_action(arm_action)
            finger_action *= 0.1
        elif phase_name == 'CONTACT':
            arm_action *= 0.05
            finger_action *= 0.2
        elif phase_name == 'GRASP':
            arm_action *= 0.02
            finger_action = self._controlled_grasp_action(finger_action)
        elif phase_name == 'LIFT':
            arm_action = self._lifting_action(arm_action)
            finger_action *= 0.01
        else:  # HOLD
            arm_action *= 0.01
            finger_action *= 0.01
        
        # Application finale avec damping
        self._apply_action_with_damping(arm_action, finger_action)
    
    def _guided_approach_action(self, arm_action):
        """Action guidée pour approcher le cube"""
        cube_pos = self._get_cube_position()
        left_hand_pos = self.data.site_xpos[0] if self.model.nsite > 0 else np.array([0, 0, 0])
        direction = cube_pos - left_hand_pos
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-6)
        approach_factor = 0.3
        arm_action[:3] += direction_normalized * approach_factor
        return np.clip(arm_action, -0.5, 0.5)
    
    def _controlled_grasp_action(self, finger_action):
        """Action contrôlée pour fermer les doigts"""
        grasp_strength = min(1.0, self.phase_timer / 30.0)
        controlled_action = np.ones_like(finger_action) * grasp_strength * 0.8
        return controlled_action
    
    def _lifting_action(self, arm_action):
        """Action pour soulever le cube"""
        lift_action = np.zeros_like(arm_action)
        lift_action[0] = 0.2
        lift_action[3] = -0.3
        lift_action[7] = 0.2
        lift_action[10] = -0.3
        return lift_action
    
    def _apply_action_with_damping(self, arm_action, finger_action):
        """Applique l'action avec damping adaptatif"""
        arm_damping = 0.8
        finger_damping = 0.5
        
        # Application aux bras (DOFs 1-14 -> actuateurs 0-13)
        for i, dof in enumerate(self.arm_dofs):
            if i < len(arm_action) and i < self.model.nu:
                target_pos = self.data.qpos[dof] + arm_action[i] * 0.05
                current_pos = self.data.qpos[dof]
                self.data.ctrl[i] = current_pos + (target_pos - current_pos) * arm_damping
        
        # Application aux doigts (DOFs 15-30 -> actuateurs 14-29)
        for i, dof in enumerate(self.finger_dofs):
            if i < len(finger_action):
                ctrl_idx = len(self.arm_dofs) + i
                if ctrl_idx < self.model.nu:
                    target_pos = self.data.qpos[dof] + finger_action[i] * 0.03
                    current_pos = self.data.qpos[dof]
                    self.data.ctrl[ctrl_idx] = current_pos + (target_pos - current_pos) * finger_damping
    
    def _update_phase(self):
        """Met à jour la phase de grasping"""
        self.phase_timer += 1
        phase_name = list(self.PHASES.keys())[self.current_phase]
        
        should_advance = False
        
        if phase_name == 'STABILIZE':
            if self.phase_timer > 20 and self._arms_are_stable():
                should_advance = True
        elif phase_name == 'APPROACH':
            if self._near_cube() or self.phase_timer > self.phase_durations[phase_name]:
                should_advance = True
        elif phase_name == 'CONTACT':
            if self._palm_cube_contact() or self.phase_timer > self.phase_durations[phase_name]:
                should_advance = True
        elif phase_name == 'GRASP':
            if self._grasp_established() or self.phase_timer > self.phase_durations[phase_name]:
                should_advance = True
        elif phase_name == 'LIFT':
            if self._cube_lifted() or self.phase_timer > self.phase_durations[phase_name]:
                should_advance = True
        
        if should_advance and self.current_phase < len(self.PHASES) - 1:
            self.episode_metrics['phase_completions'][phase_name] = 1
            self.current_phase += 1
            self.phase_timer = 0
            print(f"🎯 Phase avancée: {list(self.PHASES.keys())[self.current_phase]}")
    
    def _arms_are_stable(self) -> bool:
        """Vérifie si les bras sont stables"""
        arm_velocities = [abs(self.data.qvel[dof]) for dof in self.arm_dofs]
        return max(arm_velocities) < 0.1
    
    def _near_cube(self) -> bool:
        """Vérifie si proche du cube"""
        cube_pos = self._get_cube_position()
        if self.model.nsite > 0:
            hand_pos = self.data.site_xpos[0]
            distance = np.linalg.norm(cube_pos - hand_pos)
            return distance < 0.15
        return False
    
    def _palm_cube_contact(self) -> bool:
        """Vérifie le contact palm-cube"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            
            if (geom1_name and geom2_name and 
                (('palm' in geom1_name.lower() and 'cube' in geom2_name.lower()) or
                ('cube' in geom1_name.lower() and 'palm' in geom2_name.lower()))):
                self.episode_metrics['contact_detections'] += 1
                return True
                
        return False
    
    def _grasp_established(self) -> bool:
        """Vérifie si la prise est établie"""
        touch_contacts = 0
        for sensor_idx in self.touch_sensors:
            if abs(self.data.sensordata[sensor_idx]) > 0.01:
                touch_contacts += 1
        return touch_contacts >= 2
    
    def _cube_lifted(self) -> bool:
        """Vérifie si le cube est soulevé"""
        cube_pos = self._get_cube_position()
        initial_height = 0.05
        current_height = cube_pos[2]
        
        lifted = current_height > initial_height + 0.02
        if lifted:
            self.episode_metrics['cube_height_max'] = max(
                self.episode_metrics['cube_height_max'], 
                current_height - initial_height
            )
        
        return lifted
    
    def _get_cube_position(self) -> np.ndarray:
        """Obtient la position du cube"""
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        if cube_body_id >= 0:
            return self.data.xpos[cube_body_id].copy()
        return np.array([0.5, 0.0, 0.05])
    
    def _compute_reward(self) -> float:
        """Calcule la récompense"""
        reward = 0.0
        phase_name = list(self.PHASES.keys())[self.current_phase]
        
        if self._arms_are_stable():
            reward += 1.0
            self.episode_metrics['arm_stability_score'] += 1
        
        if phase_name == 'APPROACH':
            if self._near_cube():
                reward += 5.0
        elif phase_name == 'CONTACT':
            if self._palm_cube_contact():
                reward += 10.0
                self.episode_metrics['palm_contact_score'] += 1
        elif phase_name == 'GRASP':
            if self._grasp_established():
                reward += 15.0
                self.episode_metrics['successful_grasps'] += 1
        elif phase_name == 'LIFT':
            if self._cube_lifted():
                reward += 20.0
        elif phase_name == 'HOLD':
            if self._cube_lifted() and self._grasp_established():
                reward += 25.0
        
        if self._check_instability():
            reward -= 10.0
        
        reward += self.current_phase * 2.0
        
        return reward
    
    def _check_instability(self) -> bool:
        """Vérifie l'instabilité"""
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
            np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            return True
        
        max_velocity = np.max(np.abs(self.data.qvel))
        if max_velocity > 50.0:
            return True
            
        return False
    
    def _handle_instability(self):
        """Gère l'instabilité"""
        print("⚠️ Instabilité détectée - récupération...")
        self.data.qvel[:] = 0.0
        observation = np.zeros(self.observation_space.shape[0])
        reward = -50.0
        terminated = True
        truncated = False
        info = {'instability': True}
        return observation, reward, terminated, truncated, info
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Vérifie les conditions de fin"""
        terminated = False
        truncated = False
        
        if (self.current_phase == self.PHASES['HOLD'] and 
            self._cube_lifted() and self._grasp_established()):
            terminated = True
            
        if self._check_instability():
            terminated = True
            
        total_time = sum(self.phase_durations.values())
        if sum(self.episode_metrics['phase_completions'].values()) * 50 > total_time * 1.5:
            truncated = True
        
        return terminated, truncated
    
    def _get_observation(self) -> np.ndarray:
        """Génère l'observation"""
        obs = []
        
        # Positions des joints
        for dof in self.arm_dofs + self.finger_dofs:
            obs.append(self.data.qpos[dof])
        
        # Vitesses des joints  
        for dof in self.arm_dofs + self.finger_dofs:
            obs.append(self.data.qvel[dof])
        
        # Position/orientation du cube
        cube_pos = self._get_cube_position()
        obs.extend(cube_pos)
        
        # Orientation du cube (quaternion)
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        if cube_body_id >= 0:
            cube_quat = self.data.xquat[cube_body_id]
            obs.extend(cube_quat)
        else:
            obs.extend([1, 0, 0, 0])
        
        # Capteurs tactiles
        for sensor_idx in self.touch_sensors:
            obs.append(self.data.sensordata[sensor_idx])
        
        # Informations de phase
        obs.append(self.current_phase / len(self.PHASES))
        obs.append(self.phase_timer / 100.0)
        
        # Métriques normalisées
        obs.append(self.episode_metrics['arm_stability_score'] / 100.0)
        obs.append(self.episode_metrics['palm_contact_score'] / 10.0)
        obs.append(self.episode_metrics['successful_grasps'] / 5.0)
        obs.append(self.episode_metrics['cube_height_max'] / 0.1)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Génère les informations"""
        phase_name = list(self.PHASES.keys())[self.current_phase]
        
        return {
            'current_phase': phase_name,
            'phase_timer': self.phase_timer,
            'metrics': self.episode_metrics.copy(),
            'cube_position': self._get_cube_position().tolist(),
            'arms_stable': self._arms_are_stable(),
            'near_cube': self._near_cube(),
            'palm_contact': self._palm_cube_contact(),
            'grasp_established': self._grasp_established(),
            'cube_lifted': self._cube_lifted()
        }
    
    def render(self):
        """Rendu de l'environnement"""
        if self.render_mode == 'rgb_array' and hasattr(self, 'renderer'):
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None
    
    def _record_frame(self):
        """Enregistre une frame pour la vidéo"""
        if hasattr(self, 'renderer'):
            frame = self.render()
            if frame is not None:
                self.video_frames.append(frame)
    
    def save_video(self, filepath: str):
        """Sauvegarde la vidéo"""
        if not self.video_frames:
            return False
            
        try:
            height, width = self.video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
            
            for frame in self.video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"📹 Vidéo sauvegardée: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde vidéo: {e}")
            return False
    
    def close(self):
        """Fermeture de l'environnement"""
        if hasattr(self, 'temp_file'):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass
        
        if hasattr(self, 'renderer'):
            self.renderer.close()
            
        print("🏆 ProfessionalGraspEnv fermé")

if __name__ == "__main__":
    print("🧪 Test ProfessionalGraspEnv...")
    
    env = ProfessionalGraspEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    print(f"✅ Observation shape: {obs.shape}")
    print(f"✅ Action space: {env.action_space}")
    print(f"✅ Phase initiale: {info['current_phase']}")
    
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Reward: {reward:.2f}")
    print(f"✅ Phase: {info['current_phase']}")
    print(f"✅ Métriques: {info['metrics']}")
    
    env.close()
    print("🏆 Test réussi!")
