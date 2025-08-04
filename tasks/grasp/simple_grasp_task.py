#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasks/grasp/simple_grasp_task.py

Tâche de grasping simplifiée pour robot avec main.
Objectif : Détecter un cube, établir le contact, puis fermer les doigts pour le saisir.
"""

import numpy as np
import mujoco
import cv2
import os
from datetime import datetime

class SimpleGraspTask:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        """
        Initialise la tâche de grasping simplifiée
        
        Args:
            model: Modèle MuJoCo
            data: Données MuJoCo
            config: Configuration de la tâche
        """
        # Références MuJoCo
        self.model = model
        self.data = data
        
        # Paramètres de la tâche
        self.cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.get("cube_body_name", "cube"))
        self.max_steps = int(config.get("max_steps_per_episode", 1000))
        self.step_count = 0
        
        # Capteurs tactiles pour détecter le contact
        self.touch_sensors = config.get("touch_sensors", [])
        self.touch_ids = []
        for sensor_name in self.touch_sensors:
            try:
                sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                self.touch_ids.append(sensor_id)
            except:
                print(f"Capteur tactile {sensor_name} non trouvé")
        
        # État de la tâche
        self.contact_detected = False
        self.grasp_initiated = False
        self.grasp_completed = False
        
        # Récompenses
        self.contact_reward = 10.0
        self.grasp_reward = 50.0
        self.lift_reward_weight = 1.0
        
        # Enregistrement vidéo
        self.record_video = config.get("record_video", True)
        self.video_frames = []
        self.video_path = config.get("video_path", "videos")
        os.makedirs(self.video_path, exist_ok=True)
        
        # Dimensions d'observation et d'action
        self.obs_dim = (model.nq + model.nv + 3 + len(self.touch_ids))
        self.act_dim = model.nu
        
        # Reset initial
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset de l'environnement et de la tâche
        
        Returns:
            np.ndarray: Observation initiale
        """
        # Reset des données MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset des compteurs et états
        self.step_count = 0
        self.contact_detected = False
        self.grasp_initiated = False
        self.grasp_completed = False
        
        # Reset des frames vidéo
        self.video_frames = []
        
        return self._get_obs()
    
    def step(self, action: np.ndarray):
        """
        Exécute une étape de la simulation
        
        Args:
            action: Action du robot
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Incrémenter le compteur d'étapes
        self.step_count += 1
        
        # Appliquer l'action
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        # Obtenir l'observation
        obs = self._get_obs()
        
        # Calculer la récompense
        reward = self._compute_reward()
        
        # Vérifier si l'épisode est terminé
        done = (self.step_count >= self.max_steps) or self.grasp_completed
        
        # Enregistrer la frame si nécessaire
        if self.record_video:
            self._record_frame()
        
        return obs, reward, done, {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Construit l'observation
        
        Returns:
            np.ndarray: Vecteur d'observation
        """
        # Positions et vitesses des joints
        qpos = self.data.qpos[:self.model.nq].copy()
        qvel = self.data.qvel[:self.model.nv].copy()
        
        # Position du cube
        cube_pos = self.data.xpos[self.cube_id].copy()
        
        # Valeurs des capteurs tactiles
        touch_values = np.array([self.data.sensordata[i] for i in self.touch_ids], dtype=np.float32)
        
        # Concaténer toutes les observations
        obs = np.concatenate([qpos, qvel, cube_pos, touch_values])
        
        return obs
    
    def _compute_reward(self) -> float:
        """
        Calcule la récompense basée sur l'état actuel
        
        Returns:
            float: Récompense
        """
        reward = 0.0
        
        # Détecter le contact
        touch_values = [self.data.sensordata[i] for i in self.touch_ids]
        has_contact = any(v > 0.1 for v in touch_values)  # Seuil de contact
        
        # Récompense pour le premier contact
        if has_contact and not self.contact_detected:
            reward += self.contact_reward
            self.contact_detected = True
            print("Contact détecté!")
        
        # Récompense pour le grasping (quand les doigts sont fermés après contact)
        if self.contact_detected and not self.grasp_initiated:
            # Vérifier si les doigts sont fermés (positions des joints des doigts)
            finger_joints = self.data.qpos[-6:]  # Supposons 6 joints pour les doigts
            fingers_closed = all(joint < -0.5 for joint in finger_joints)
            
            if fingers_closed:
                reward += self.grasp_reward
                self.grasp_initiated = True
                print("Grasping initié!")
        
        # Récompense pour soulever le cube
        if self.grasp_initiated:
            cube_height = self.data.xpos[self.cube_id][2]
            reward += self.lift_reward_weight * cube_height
            
            # Vérifier si le cube est bien saisi (ne tombe pas)
            if cube_height > 0.1:  # 10cm au-dessus de la table
                self.grasp_completed = True
                print("Grasping réussi!")
        
        return reward
    
    def _record_frame(self):
        """
        Enregistre la frame actuelle pour la vidéo
        """
        try:
            # Créer un renderer temporaire
            from mujoco import Renderer
            renderer = Renderer(self.model, width=640, height=480)
            renderer.update_scene(self.data)
            frame = renderer.render()
            
            # Convertir BGR vers RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_frames.append(frame_rgb)
            
        except Exception as e:
            print(f"Erreur lors de l'enregistrement de la frame: {e}")
    
    def save_video(self, episode_num: int):
        """
        Sauvegarde la vidéo de l'épisode
        
        Args:
            episode_num: Numéro de l'épisode
        """
        if not self.record_video or not self.video_frames:
            return
        
        try:
            # Créer le nom du fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grasp_episode_{episode_num}_{timestamp}.mp4"
            filepath = os.path.join(self.video_path, filename)
            
            # Paramètres de la vidéo
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            height, width = self.video_frames[0].shape[:2]
            
            # Créer le writer vidéo
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            # Écrire toutes les frames
            for frame in self.video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Vidéo sauvegardée: {filepath}")
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la vidéo: {e}")
    
    def get_task_info(self) -> dict:
        """
        Retourne les informations sur l'état de la tâche
        
        Returns:
            dict: Informations sur la tâche
        """
        return {
            "step_count": self.step_count,
            "contact_detected": self.contact_detected,
            "grasp_initiated": self.grasp_initiated,
            "grasp_completed": self.grasp_completed,
            "cube_height": float(self.data.xpos[self.cube_id][2]),
            "touch_values": [float(self.data.sensordata[i]) for i in self.touch_ids]
        }