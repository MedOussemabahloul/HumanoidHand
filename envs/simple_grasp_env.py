#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
envs/simple_grasp_env.py

Environnement simplifié pour l'apprentissage du grasping.
Interface Gymnasium avec MuJoCo pour la simulation du robot et du cube.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
from mujoco import Renderer
import os

class SimpleGraspEnv(gym.Env):
    """
    Environnement simplifié pour l'apprentissage du grasping
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 xml_path: str = "assets/scenes/complete_scene.xml",
                 render_mode: str = None,
                 width: int = 640,
                 height: int = 480,
                 config: dict = None):
        """
        Initialise l'environnement
        
        Args:
            xml_path: Chemin vers le fichier XML MuJoCo
            render_mode: Mode de rendu ("human", "rgb_array", None)
            width: Largeur de la fenêtre de rendu
            height: Hauteur de la fenêtre de rendu
            config: Configuration de l'environnement
        """
        super().__init__()
        
        # Configuration par défaut
        self.config = config or {}
        
        # Charger le modèle MuJoCo
        print(f"[DEBUG] Chargement du modèle MuJoCo depuis: {xml_path}")
        try:
            self.model = MjModel.from_xml_path(xml_path)
            self.data = MjData(self.model)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            # Créer un modèle simple par défaut si le fichier n'existe pas
            self._create_simple_model()
        
        # Paramètres de rendu
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.viewer = None
        self.renderer = None
        
        # Espaces d'action et d'observation
        n_act = self.model.nu
        n_obs = self.model.nq + self.model.nv + 3  # qpos + qvel + cube_pos
        
        # Ajouter les capteurs tactiles si disponibles
        touch_sensors = self.config.get("touch_sensors", [])
        for sensor_name in touch_sensors:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                n_obs += 1
            except:
                print(f"Capteur tactile {sensor_name} non trouvé")
        
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        
        # ID du cube
        try:
            self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 
                                           self.config.get("cube_body_name", "cube"))
        except:
            print("Cube non trouvé dans le modèle")
            self.cube_id = None
        
        # Capteurs tactiles
        self.touch_ids = []
        for sensor_name in touch_sensors:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                self.touch_ids.append(sensor_id)
            except:
                print(f"Capteur tactile {sensor_name} non trouvé")
        
        # État de l'environnement
        self.step_count = 0
        self.max_steps = self.config.get("max_steps_per_episode", 1000)
        
        # Initialisation
        mj_forward(self.model, self.data)
    
    def _create_simple_model(self):
        """
        Crée un modèle MuJoCo simple par défaut si le fichier XML n'existe pas
        """
        print("Création d'un modèle simple par défaut...")
        
        # XML simple avec un robot basique et un cube
        simple_xml = """
        <mujoco model="simple_grasp">
            <option timestep="0.002" iterations="50" solver="Newton" tolerance="1e-10"/>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
                <material name="robot" rgba="0.7 0.7 0.7 1"/>
                <material name="cube" rgba="0.8 0.2 0.2 1"/>
            </asset>
            
            <worldbody>
                <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
                <geom name="ground" type="plane" pos="0 0 0" size="0 0 .05" material="grid"/>
                
                <!-- Robot simple avec main -->
                <body name="robot_base" pos="0 0 0.5">
                    <geom type="cylinder" size="0.1 0.2" material="robot"/>
                    <joint type="free"/>
                    
                    <!-- Bras -->
                    <body name="arm" pos="0 0 0.3">
                        <geom type="cylinder" size="0.05 0.3" material="robot"/>
                        <joint name="arm_joint" type="slide" axis="0 0 1" range="-0.5 0.5"/>
                        
                        <!-- Main avec doigts -->
                        <body name="hand" pos="0 0 0.3">
                            <geom type="sphere" size="0.08" material="robot"/>
                            <joint name="hand_joint" type="ball" pos="0 0 0"/>
                            
                            <!-- Doigts -->
                            <body name="finger1" pos="0.05 0 0">
                                <geom type="capsule" size="0.02 0.1" material="robot"/>
                                <joint name="finger1_joint" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                                <site name="touch1" pos="0 0 0.1" size="0.01" rgba="1 0 0 1"/>
                            </body>
                            
                            <body name="finger2" pos="-0.05 0 0">
                                <geom type="capsule" size="0.02 0.1" material="robot"/>
                                <joint name="finger2_joint" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                                <site name="touch2" pos="0 0 0.1" size="0.01" rgba="1 0 0 1"/>
                            </body>
                        </body>
                    </body>
                </body>
                
                <!-- Cube à saisir -->
                <body name="cube" pos="0.2 0 0.05">
                    <geom type="box" size="0.05 0.05 0.05" material="cube"/>
                    <joint type="free"/>
                </body>
            </worldbody>
            
            <sensor>
                <touch name="touch1_sensor" site="touch1"/>
                <touch name="touch2_sensor" site="touch2"/>
            </sensor>
            
            <actuator>
                <motor name="arm_motor" joint="arm_joint" gear="100"/>
                <motor name="finger1_motor" joint="finger1_joint" gear="50"/>
                <motor name="finger2_motor" joint="finger2_joint" gear="50"/>
            </actuator>
        </mujoco>
        """
        
        # Créer le modèle à partir du XML
        self.model = MjModel.from_xml_string(simple_xml)
        self.data = MjData(self.model)
        
        # Mettre à jour les IDs
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.touch_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch1_sensor"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch2_sensor")
        ]
    
    def reset(self, *, seed=None, options=None):
        """
        Reset de l'environnement
        
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
            cube_pos[0] = np.random.uniform(-0.3, 0.3)  # X aléatoire
            cube_pos[1] = np.random.uniform(-0.3, 0.3)  # Y aléatoire
            cube_pos[2] = 0.05  # Z fixe sur la table
        
        # Forward pour mettre à jour les positions
        mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Exécute une étape de simulation
        
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
        
        # Info
        info = {
            'step_count': self.step_count,
            'cube_height': float(self.data.xpos[self.cube_id][2]) if self.cube_id else 0.0,
            'touch_values': [float(self.data.sensordata[i]) for i in self.touch_ids]
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_obs(self):
        """
        Construit l'observation
        
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
        
        # Valeurs des capteurs tactiles
        touch_values = np.array([self.data.sensordata[i] for i in self.touch_ids], dtype=np.float32)
        
        # Concaténer toutes les observations
        obs = np.concatenate([qpos, qvel, cube_pos, touch_values])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        """
        Calcule la récompense basée sur l'état actuel
        
        Returns:
            float: Récompense
        """
        reward = 0.0
        
        # Récompense pour le contact
        touch_values = [self.data.sensordata[i] for i in self.touch_ids]
        has_contact = any(v > 0.1 for v in touch_values)
        if has_contact:
            reward += 1.0
        
        # Récompense pour soulever le cube
        if self.cube_id is not None:
            cube_height = self.data.xpos[self.cube_id][2]
            reward += cube_height * 0.1  # Récompense proportionnelle à la hauteur
        
        # Pénalité pour les actions trop grandes
        action_penalty = -0.01 * np.mean(np.square(self.data.ctrl))
        reward += action_penalty
        
        return reward
    
    def render(self, mode=None):
        """
        Rend l'environnement
        
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
        Ferme l'environnement
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer = None
    
    def get_task_info(self):
        """
        Retourne les informations sur l'état de la tâche
        
        Returns:
            dict: Informations sur la tâche
        """
        return {
            'step_count': self.step_count,
            'cube_height': float(self.data.xpos[self.cube_id][2]) if self.cube_id else 0.0,
            'touch_values': [float(self.data.sensordata[i]) for i in self.touch_ids],
            'robot_pos': self.data.qpos[:3].copy() if len(self.data.qpos) >= 3 else np.zeros(3)
        }