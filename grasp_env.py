#!/usr/bin/env python3
"""
🤖 ENVIRONNEMENT DE GRASPING AVEC G1_COMBINED
=============================================

Environnement simple et robuste pour l'apprentissage de grasping:
- Utilise le modèle g1_combined.xml existant
- Cube fixe sur la table
- Détection de contact physique réaliste
- Contrôle de force adaptatif
- Dimensions d'observation correctes (88D)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os
from typing import Dict, Any
import cv2
import imageio

class GraspEnv(gym.Env):
  """Environnement de grasping utilisant g1_combined.xml"""
  
  def __init__(self, render_mode="rgb_array", record_video=False, video_dir="videos"):
      super().__init__()
      
      self.render_mode = render_mode
      self.record_video = record_video
      self.video_dir = video_dir
      self.episode_count = 0
      
      # Créer le dossier vidéo seulement si on enregistre des vidéos
      if record_video:
          os.makedirs(video_dir, exist_ok=True)
      
      # Charger le modèle g1_combined.xml
      model_path = "results/g1_combined.xml"
      if not os.path.exists(model_path):
          raise FileNotFoundError(f"Modèle g1_combined.xml non trouvé: {model_path}")
      
      self.model = mujoco.MjModel.from_xml_path(model_path)
      self.data = mujoco.MjData(self.model)
      
      # Identifier les composants
      self._identify_components()
      
      # Configuration des espaces
      # Actions: 14 joints bras + 8 joints doigts = 22 actions
      self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(22,), dtype=np.float32)
      
      # Observations: 88 dimensions exactes
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32)
      
      # Phases de grasping
      self.PHASES = ['SEARCH', 'APPROACH', 'CONTACT', 'ALIGN', 'GRASP', 'LIFT', 'HOLD']
      self.current_phase = 0
      self.phase_timer = 0
      self.episode_step = 0
      self.max_episode_steps = 500
      
      # Métriques
      self.contact_detected = False
      self.cube_grasped = False
      self.cube_lifted = False
      self.grasp_force = 0.0
      
      # Vidéo
      self.video_frames = []
      
      print(f"✅ GraspEnv initialisé avec g1_combined.xml")
      print(f"  📐 Actions: {self.action_space.shape}")
      print(f"  👁️  Observations: {self.observation_space.shape}")
      print(f"  🎬 Vidéo: {'✅' if record_video else '❌'}")
  
  def _identify_components(self):
      """Identifie les joints et corps du modèle"""
      
      # Joints des bras (14 joints)
      arm_joint_names = [
          'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
          'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
          'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
          'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
      ]
      
      # Joints des doigts (8 joints principaux)
      finger_joint_names = [
          'left_index_joint_0', 'left_middle_joint_0', 'left_ring_joint_0', 'left_thumb_joint_0',
          'right_index_joint_0', 'right_middle_joint_0', 'right_ring_joint_0', 'right_thumb_joint_0'
      ]
      
      self.arm_joint_ids = []
      for name in arm_joint_names:
          joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
          if joint_id >= 0:
              self.arm_joint_ids.append(joint_id)
      
      self.finger_joint_ids = []
      for name in finger_joint_names:
          joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
          if joint_id >= 0:
              self.finger_joint_ids.append(joint_id)
      
      # Corps du cube
      self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
      
      print(f"  🔧 Joints bras trouvés: {len(self.arm_joint_ids)}")
      print(f"  🔧 Joints doigts trouvés: {len(self.finger_joint_ids)}")
      print(f"  📦 Cube ID: {self.cube_body_id}")
  
  def reset(self, seed=None, options=None):
      """Réinitialise l'environnement"""
      super().reset(seed=seed)
      
      if seed is not None:
          np.random.seed(seed)
      
      # Réinitialiser la simulation
      mujoco.mj_resetData(self.model, self.data)
      
      # Position initiale des bras
      for i, joint_id in enumerate(self.arm_joint_ids):
          if i < 7:  # Bras gauche
              self.data.qpos[joint_id] = np.random.uniform(-0.1, 0.1)
          else:  # Bras droit
              self.data.qpos[joint_id] = np.random.uniform(-0.1, 0.1)
      
      # Position initiale des doigts (légèrement ouverts)
      for joint_id in self.finger_joint_ids:
          self.data.qpos[joint_id] = np.random.uniform(0.0, 0.3)
      
      # Réinitialiser les métriques
      self.current_phase = 0
      self.phase_timer = 0
      self.episode_step = 0
      self.contact_detected = False
      self.cube_grasped = False
      self.cube_lifted = False
      self.grasp_force = 0.0
      
      # Vider les frames vidéo
      self.video_frames = []
      
      # Stabiliser la simulation
      for _ in range(10):
          mujoco.mj_step(self.model, self.data)
      
      observation = self._get_observation()
      info = self._get_info()
      
      return observation, info
  
  def step(self, action):
      """Exécute une action"""
      
      action = np.clip(action, -1.0, 1.0)
      
      # Appliquer les actions
      self._apply_action(action)
      
      # Simulation physique
      mujoco.mj_step(self.model, self.data)
      
      # Mettre à jour les métriques
      self._update_metrics()
      
      # Calculer la récompense
      reward = self._compute_reward()
      
      # Vérifier la terminaison
      terminated = self._check_termination()
      truncated = self.episode_step >= self.max_episode_steps
      
      # Enregistrer la frame vidéo
      if self.record_video:
          self._record_frame()
      
      self.episode_step += 1
      self.phase_timer += 1
      
      observation = self._get_observation()
      info = self._get_info()
      
      return observation, reward, terminated, truncated, info
  
  def _apply_action(self, action):
      """Applique l'action aux articulations"""
      
      # Actions des bras (14 premières)
      arm_actions = action[:14]
      for i, joint_id in enumerate(self.arm_joint_ids):
          if i < len(arm_actions):
              current_pos = self.data.qpos[joint_id]
              target_pos = current_pos + arm_actions[i] * 0.1  # Gain contrôlé
              # Appliquer les limites
              joint_range = self.model.jnt_range[joint_id]
              if joint_range[0] < joint_range[1]:
                  target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
              self.data.ctrl[i] = target_pos
      
      # Actions des doigts (8 dernières)
      finger_actions = action[14:22]
      for i, joint_id in enumerate(self.finger_joint_ids):
          if i < len(finger_actions):
              current_pos = self.data.qpos[joint_id]
              target_pos = current_pos + finger_actions[i] * 0.05  # Mouvement plus fin
              target_pos = np.clip(target_pos, 0.0, 1.5)  # Limites doigts
              ctrl_idx = len(self.arm_joint_ids) + i
              if ctrl_idx < self.model.nu:
                  self.data.ctrl[ctrl_idx] = target_pos
  
  def _update_metrics(self):
      """Met à jour les métriques de performance"""
      
      # Position du cube
      cube_pos = self._get_cube_position()
      
      # Positions des mains
      left_hand_pos = self._get_hand_position("left")
      right_hand_pos = self._get_hand_position("right")
      
      # Distance au cube
      left_dist = np.linalg.norm(cube_pos - left_hand_pos)
      right_dist = np.linalg.norm(cube_pos - right_hand_pos)
      min_distance = min(left_dist, right_dist)
      
      # Détecter le contact (approximation)
      self.contact_detected = min_distance < 0.1
      
      # Cube saisi (approximation basée sur distance et force)
      finger_positions = [self.data.qpos[joint_id] for joint_id in self.finger_joint_ids]
      finger_closure = np.mean(finger_positions)
      self.cube_grasped = self.contact_detected and finger_closure > 0.5
      
      # Cube levé (vérifie la hauteur)
      initial_cube_height = 0.055  # Position initiale du cube
      self.cube_lifted = cube_pos[2] > initial_cube_height + 0.05 and self.cube_grasped
      
      # Force de saisie
      self.grasp_force = finger_closure / 1.5 if self.contact_detected else 0.0
      
      # Progression des phases
      self._update_phase_progression(min_distance)
  
  def _update_phase_progression(self, min_distance):
      """Met à jour la progression des phases"""
      
      # Progression automatique selon les métriques
      if self.current_phase == 0 and min_distance < 0.3:  # SEARCH -> APPROACH
          self.current_phase = 1
          self.phase_timer = 0
      elif self.current_phase == 1 and min_distance < 0.15:  # APPROACH -> CONTACT
          self.current_phase = 2
          self.phase_timer = 0
      elif self.current_phase == 2 and self.contact_detected:  # CONTACT -> ALIGN
          self.current_phase = 3
          self.phase_timer = 0
      elif self.current_phase == 3 and self.grasp_force > 0.3:  # ALIGN -> GRASP
          self.current_phase = 4
          self.phase_timer = 0
      elif self.current_phase == 4 and self.cube_grasped:  # GRASP -> LIFT
          self.current_phase = 5
          self.phase_timer = 0
      elif self.current_phase == 5 and self.cube_lifted:  # LIFT -> HOLD
          self.current_phase = 6
          self.phase_timer = 0
  
  def _compute_reward(self):
      """Calcule la récompense"""
      
      reward = 0.0
      cube_pos = self._get_cube_position()
      
      # Distance au cube
      left_hand_pos = self._get_hand_position("left")
      right_hand_pos = self._get_hand_position("right")
      left_dist = np.linalg.norm(cube_pos - left_hand_pos)
      right_dist = np.linalg.norm(cube_pos - right_hand_pos)
      min_distance = min(left_dist, right_dist)
      
      # Récompenses par phase
      if self.current_phase == 0:  # SEARCH
          reward += 1.0 - min(min_distance / 0.5, 1.0)
      elif self.current_phase == 1:  # APPROACH
          reward += 2.0 - min(min_distance / 0.2, 2.0)
      elif self.current_phase == 2:  # CONTACT
          reward += 5.0 if self.contact_detected else 0.0
      elif self.current_phase == 3:  # ALIGN
          reward += 8.0 * self.grasp_force
      elif self.current_phase == 4:  # GRASP
          reward += 15.0 if self.cube_grasped else 0.0
      elif self.current_phase == 5:  # LIFT
          height_bonus = max(0, cube_pos[2] - 0.055)
          reward += 25.0 * min(height_bonus / 0.1, 1.0)
      elif self.current_phase == 6:  # HOLD
          if self.cube_lifted:
              reward += 50.0
      
      # Bonus de progression
      reward += self.current_phase * 3.0
      
      # Pénalités
      if min_distance > 0.8:  # Trop loin du cube
          reward -= 1.0
      
      return reward
  
  def _get_observation(self):
      """Construit l'observation de 88 dimensions"""
      
      obs = []
      
      # Positions des joints (36 valeurs)
      joint_positions = self.data.qpos.copy()
      if len(joint_positions) >= 36:
          obs.extend(joint_positions[:36])
      else:
          # Pad avec des zéros si nécessaire
          padded = np.zeros(36)
          padded[:len(joint_positions)] = joint_positions
          obs.extend(padded)
      
      # Vitesses des joints (36 valeurs)
      joint_velocities = self.data.qvel.copy()
      if len(joint_velocities) >= 36:
          obs.extend(joint_velocities[:36])
      else:
          # Pad avec des zéros si nécessaire
          padded = np.zeros(36)
          padded[:len(joint_velocities)] = joint_velocities
          obs.extend(padded)
      
      # Position du cube (3 valeurs)
      cube_pos = self._get_cube_position()
      obs.extend(cube_pos)
      
      # Orientation du cube (4 valeurs - quaternion)
      if self.cube_body_id >= 0:
          cube_quat = self.data.xquat[self.cube_body_id].copy()
          obs.extend(cube_quat)
      else:
          obs.extend([1.0, 0.0, 0.0, 0.0])
      
      # Vitesse du cube (3 valeurs)
      cube_vel = self._get_cube_velocity()
      obs.extend(cube_vel)
      
      # Métriques de grasping (6 valeurs)
      obs.append(float(self.current_phase))
      obs.append(float(self.contact_detected))
      obs.append(float(self.cube_grasped))
      obs.append(float(self.cube_lifted))
      obs.append(float(self.grasp_force))
      obs.append(float(self.phase_timer / 100.0))  # Timer normalisé
      
      # S'assurer qu'on a exactement 88 dimensions
      obs_array = np.array(obs, dtype=np.float32)
      if len(obs_array) < 88:
          final_obs = np.zeros(88, dtype=np.float32)
          final_obs[:len(obs_array)] = obs_array
          return final_obs
      else:
          return obs_array[:88]
  
  def _get_cube_position(self):
      """Obtient la position du cube"""
      if self.cube_body_id >= 0:
          return self.data.xpos[self.cube_body_id].copy()
      else:
          return np.array([0.3, 0.0, 0.055])
  
  def _get_cube_velocity(self):
      """Obtient la vitesse du cube"""
      if self.cube_body_id >= 0:
          return self.data.cvel[self.cube_body_id][:3].copy()
      else:
          return np.array([0.0, 0.0, 0.0])
  
  def _get_hand_position(self, side):
      """Obtient la position d'une main"""
      # Approximation basée sur le poignet
      wrist_joint = f"{side}_wrist_yaw_joint"
      joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, wrist_joint)
      if joint_id >= 0:
          # Position approximative de la main basée sur le poignet
          return self.data.xpos[joint_id] if joint_id < len(self.data.xpos) else np.array([0.0, 0.0, 1.0])
      else:
          return np.array([0.0, 0.2 if side == "left" else -0.2, 1.0])
  
  def _check_termination(self):
      """Vérifie les conditions de terminaison"""
      
      # Succès: cube maintenu en l'air
      if self.current_phase >= 6 and self.cube_lifted and self.phase_timer > 30:
          return True
      
      # Échec: positions invalides
      if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
          return True
      
      return False
  
  def _record_frame(self):
      """Enregistre une frame vidéo"""
      if self.render_mode == "rgb_array":
          frame = self.render()
          if frame is not None:
              self.video_frames.append(frame)
  
  def _get_info(self):
      """Retourne les informations de debug"""
      return {
          'phase': self.PHASES[self.current_phase],
          'phase_timer': self.phase_timer,
          'episode_step': self.episode_step,
          'cube_position': self._get_cube_position().tolist(),
          'contact_detected': self.contact_detected,
          'cube_grasped': self.cube_grasped,
          'cube_lifted': self.cube_lifted,
          'grasp_force': self.grasp_force
      }
  
  def render(self, mode=None):
      """Rendu de l'environnement"""
      if mode is None:
          mode = self.render_mode
      
      if mode == "rgb_array":
          width, height = 640, 480
          
          try:
              # Utiliser le renderer MuJoCo
              renderer = mujoco.Renderer(self.model, width, height)
              renderer.update_scene(self.data, camera="main_camera")
              frame = renderer.render()
              renderer.close()
              return frame
          except:
              # Fallback: frame vide
              return np.zeros((height, width, 3), dtype=np.uint8)
      
      return None
  
  def save_video(self, filename=None):
      """Sauvegarde la vidéo de l'épisode"""
      if not self.video_frames:
          return
      
      # Créer le dossier vidéo si nécessaire
      os.makedirs(self.video_dir, exist_ok=True)
      
      if filename is None:
          filename = f"grasp_episode_{self.episode_count:04d}.mp4"
      
      video_path = os.path.join(self.video_dir, filename)
      
      try:
          with imageio.get_writer(video_path, fps=30) as writer:
              for frame in self.video_frames:
                  writer.append_data(frame)
          
          print(f"🎬 Vidéo sauvegardée: {video_path}")
          self.episode_count += 1
          
      except Exception as e:
          print(f"⚠️  Erreur vidéo: {e}")
  
  def close(self):
      """Ferme l'environnement"""
      if hasattr(self, 'video_frames') and self.video_frames:
          self.save_video()
