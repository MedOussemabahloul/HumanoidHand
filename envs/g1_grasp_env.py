#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
envs/g1_grasp_env.py

Environnement spécifique pour le modèle G1 avec vrais capteurs et joints.
Interface Gymnasium avec MuJoCo pour la simulation du robot G1 et du cube.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
from mujoco import Renderer
import os

class G1GraspEnv(gym.Env):
    """
    Environnement spécifique pour le modèle G1 avec vrais capteurs et joints
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 xml_path: str = "results/g1_combined.xml",
                 render_mode: str = None,
                 width: int = 640,
                 height: int = 480,
                 config: dict = None):
        """
        Initialise l'environnement G1
        
        Args:
            xml_path: Chemin vers le fichier XML MuJoCo G1
            render_mode: Mode de rendu ("human", "rgb_array", None)
            width: Largeur de la fenêtre de rendu
            height: Hauteur de la fenêtre de rendu
            config: Configuration de l'environnement
        """
        super().__init__()
        
        # Configuration par défaut
        self.config = config or {}
        
        # Charger le modèle MuJoCo G1
        print(f"[DEBUG] Chargement du modèle G1 depuis: {xml_path}")
        try:
            self.model = MjModel.from_xml_path(xml_path)
            self.data = MjData(self.model)
            print(f"✅ Modèle G1 chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle G1: {e}")
            raise
        
        # Paramètres de rendu
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.viewer = None
        self.renderer = None
        
        # ID du cube
        try:
            self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 
                                           self.config.get("cube_body_name", "cube"))
            print(f"✅ Cube trouvé: ID {self.cube_id}")
        except:
            print("❌ Cube non trouvé dans le modèle G1")
            self.cube_id = None
        
        # Capteurs de force
        self.force_sensors = self.config.get("force_sensors", [])
        self.force_ids = []
        for sensor_name in self.force_sensors:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                self.force_ids.append(sensor_id)
                print(f"✅ Capteur de force trouvé: {sensor_name} (ID: {sensor_id})")
            except:
                print(f"⚠ Capteur de force non trouvé: {sensor_name}")
        
        # Capteurs tactiles
        self.touch_sensors = self.config.get("touch_sensors", [])
        self.touch_ids = []
        for sensor_name in self.touch_sensors:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                self.touch_ids.append(sensor_id)
                print(f"✅ Capteur tactile trouvé: {sensor_name} (ID: {sensor_id})")
            except:
                print(f"⚠ Capteur tactile non trouvé: {sensor_name}")
        
        # Joints des doigts
        self.finger_joints = self.config.get("finger_joints", [])
        self.finger_joint_ids = []
        for joint_name in self.finger_joints:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.finger_joint_ids.append(joint_id)
                print(f"✅ Joint de doigt trouvé: {joint_name} (ID: {joint_id})")
            except:
                print(f"⚠ Joint de doigt non trouvé: {joint_name}")
        
        # Espaces d'action et d'observation
        n_act = self.model.nu
        n_obs = self.model.nq + self.model.nv + 3  # qpos + qvel + cube_pos
        
        # Ajouter les capteurs
        n_obs += len(self.force_ids) + len(self.touch_ids)
        
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        
        print(f"📊 Espaces définis:")
        print(f"   - Observations: {self.observation_space.shape}")
        print(f"   - Actions: {self.action_space.shape}")
        print(f"   - Capteurs de force: {len(self.force_ids)}")
        print(f"   - Capteurs tactiles: {len(self.touch_ids)}")
        print(f"   - Joints de doigts: {len(self.finger_joint_ids)}")
        
        # État de l'environnement
        self.step_count = 0
        self.max_steps = self.config.get("max_steps_per_episode", 1000)
        
        # Initialisation
        mj_forward(self.model, self.data)
    
    def reset(self, *, seed=None, options=None):
        """
        Reset de l'environnement G1
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset des données MuJoCo
        mj_resetData(self.model, self.data)
        
        # Reset des compteurs
        self.step_count = 0
        
        # Position initiale aléatoire du cube
        if self.cube_id is not None:
            cube_pos = self.data.xpos[self.cube_id]
            cube_pos[0] = np.random.uniform(0.2, 0.4)  # X aléatoire
            cube_pos[1] = np.random.uniform(-0.1, 0.1)  # Y aléatoire
            cube_pos[2] = 0.05  # Z fixe sur la table
        
        # Forward pour mettre à jour les positions
        mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Exécute une étape de simulation G1
        
        Args:
            action: Action du robot
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Incrémenter le compteur
        self.step_count += 1
        
        # Appliquer l'action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action.astype(float)
        
        # Step de simulation
        mj_step(self.model, self.data)
        
        # Obtenir l'observation
        obs = self._get_obs()
        
        # Calculer la récompense
        reward = self._compute_reward()
        
        # Vérifier si l'épisode est terminé
        terminated = False
        truncated = (self.step_count >= self.max_steps)
        
        # Info détaillée
        force_values = [float(self.data.sensordata[i]) for i in self.force_ids]
        touch_values = [float(self.data.sensordata[i]) for i in self.touch_ids]
        finger_positions = [float(self.data.qpos[i]) for i in self.finger_joint_ids] if self.finger_joint_ids else []
        
        info = {
            'step_count': self.step_count,
            'cube_height': float(self.data.xpos[self.cube_id][2]) if self.cube_id else 0.0,
            'force_values': force_values,
            'touch_values': touch_values,
            'finger_positions': finger_positions,
            'has_force_contact': any(v > 0.1 for v in force_values),
            'has_touch_contact': any(v > 0.5 for v in touch_values),
            'fingers_closed': all(pos < -0.5 for pos in finger_positions) if finger_positions else False
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_obs(self):
        """
        Construit l'observation pour le modèle G1
        
        Returns:
            np.ndarray: Vecteur d'observation
        """
        # Positions et vitesses des joints
        qpos = self.data.qpos[:self.model.nq].copy()
        qvel = self.data.qvel[:self.model.nv].copy()
        
        # Position du cube
        if self.cube_id is not None:
            cube_pos = self.data.xpos[self.cube_id].copy()
        else:
            cube_pos = np.zeros(3)
        
        # Valeurs des capteurs de force
        force_values = np.array([self.data.sensordata[i] for i in self.force_ids], dtype=np.float32)
        
        # Valeurs des capteurs tactiles
        touch_values = np.array([self.data.sensordata[i] for i in self.touch_ids], dtype=np.float32)
        
        # Concaténer toutes les observations
        obs = np.concatenate([qpos, qvel, cube_pos, force_values, touch_values])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        """
        Calcule la récompense basée sur l'état actuel du G1
        
        Returns:
            float: Récompense
        """
        reward = 0.0
        
        # Récompense pour le contact (capteurs de force)
        force_values = [self.data.sensordata[i] for i in self.force_ids]
        has_force_contact = any(v > 0.1 for v in force_values)
        if has_force_contact:
            reward += 1.0
        
        # Récompense pour le contact (capteurs tactiles)
        touch_values = [self.data.sensordata[i] for i in self.touch_ids]
        has_touch_contact = any(v > 0.5 for v in touch_values)
        if has_touch_contact:
            reward += 2.0
        
        # Récompense pour soulever le cube
        if self.cube_id is not None:
            cube_height = self.data.xpos[self.cube_id][2]
            reward += cube_height * 0.1  # Récompense proportionnelle à la hauteur
        
        # Récompense pour fermer les doigts (grasping)
        if self.finger_joint_ids:
            finger_positions = [self.data.qpos[i] for i in self.finger_joint_ids]
            fingers_closed = all(pos < -0.5 for pos in finger_positions)
            if fingers_closed and (has_force_contact or has_touch_contact):
                reward += 5.0
        
        # Pénalité pour les actions trop grandes
        action_penalty = -0.01 * np.mean(np.square(self.data.ctrl))
        reward += action_penalty
        
        return reward
    
    def render(self, mode=None):
        """
        Rend l'environnement G1
        
        Args:
            mode: Mode de rendu ("human", "rgb_array")
            
        Returns:
            np.ndarray or None: Image si mode "rgb_array", None sinon
        """
        mode = mode or self.render_mode
        
        if mode == "human":
            if self.viewer is None:
                try:
                    from mujoco.viewer import launch_passive
                    self.viewer = launch_passive(self.model, self.data)
                except ImportError:
                    print("Mujoco viewer non disponible")
                    return None
            self.viewer.sync()
            return None
            
        elif mode == "rgb_array":
            if self.renderer is None:
                self.renderer = Renderer(self.model, width=self.width, height=self.height)
            self.renderer.update_scene(self.data)
            frame = self.renderer.render()
            return frame
            
        else:
            raise ValueError(f"Mode de rendu inconnu: {mode}")
    
    def close(self):
        """
        Ferme l'environnement G1
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer = None
    
    def get_task_info(self):
        """
        Retourne les informations détaillées sur l'état de la tâche G1
        
        Returns:
            dict: Informations sur la tâche
        """
        force_values = [float(self.data.sensordata[i]) for i in self.force_ids]
        touch_values = [float(self.data.sensordata[i]) for i in self.touch_ids]
        finger_positions = [float(self.data.qpos[i]) for i in self.finger_joint_ids] if self.finger_joint_ids else []
        
        return {
            'step_count': self.step_count,
            'cube_height': float(self.data.xpos[self.cube_id][2]) if self.cube_id else 0.0,
            'force_values': force_values,
            'touch_values': touch_values,
            'finger_positions': finger_positions,
            'has_force_contact': any(v > 0.1 for v in force_values),
            'has_touch_contact': any(v > 0.5 for v in touch_values),
            'fingers_closed': all(pos < -0.5 for pos in finger_positions) if finger_positions else False
        }