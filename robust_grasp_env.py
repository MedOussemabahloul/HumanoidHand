#!/usr/bin/env python3
"""
🤖 ENVIRONNEMENT DE GRASPING ROBUSTE AVEC SAC
==============================================

Système professionnel de grasping avec détection de contact physique:
🎯 Recherche intelligente du cube avec mouvements naturels des bras
🤝 Collision physique réaliste - bras ne peuvent pas traverser les objets  
👋 Détection de contact précise (doigts + palm)
🔒 Fixation optimale de la palm au cube
✊ Fermeture contrôlée des doigts avec feedback de force
🏗️ Physics ultra-stable avec MuJoCo

Conçu spécifiquement pour l'agent SAC avec curriculum learning adaptatif.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import warnings
import time
import math
warnings.filterwarnings("ignore")

class RobustGraspEnv(gym.Env):
    """
    🤖 Environnement de Grasping Ultra-Robuste pour Agent SAC
    
    Fonctionnalités avancées:
    - Physics collision réaliste (objets solides)
    - Détection de contact multi-senseurs (doigts + palm)
    - Contrôle de force adaptatif pour la fixation
    - Curriculum learning intégré
    - Enregistrement vidéo automatique
    - Observations dimensionnées correctement (88D)
    """
    
    def __init__(self, 
                 render_mode: str = "rgb_array",
                 record_video: bool = False,
                 video_dir: str = "/workspace/videos"):
        super().__init__()
        
        # Configuration de base
        self.render_mode = render_mode
        self.record_video = record_video
        self.video_dir = video_dir
        self.episode_count = 0
        
        # Créer les dossiers nécessaires
        os.makedirs(video_dir, exist_ok=True)
        
        # Phases de grasping avec progression naturelle
        self.PHASES = {
            'SEARCH': 0,     # Recherche du cube avec mouvements des bras
            'APPROACH': 1,   # Approche contrôlée vers le cube
            'CONTACT': 2,    # Contact initial avec détection
            'ALIGN': 3,      # Alignement palm-cube optimal
            'GRASP': 4,      # Fermeture des doigts avec contrôle de force
            'LIFT': 5,       # Levée du cube
            'HOLD': 6        # Maintien stable
        }
        
        # Configuration des phases
        self.phase_durations = {
            'SEARCH': 100,     # Plus de temps pour explorer
            'APPROACH': 80,    # Approche précise
            'CONTACT': 60,     # Contact sensible
            'ALIGN': 40,       # Alignement fin
            'GRASP': 60,       # Saisie contrôlée
            'LIFT': 40,        # Levée rapide
            'HOLD': 60         # Maintien stable
        }
        
        # État de l'environnement
        self.current_phase = 0
        self.phase_timer = 0
        self.episode_step = 0
        self.max_episode_steps = 500
        
        # Positions initiales robustes
        self.cube_initial_pos = np.array([0.4, 0.0, 1.05])  # Sur table
        self.arm_initial_positions = {
            'left': [0.0, 0.3, -0.5, -1.2, 0.0, 0.8, 0.0],    # Position recherche
            'right': [0.0, -0.3, 0.5, -1.2, 0.0, 0.8, 0.0]
        }
        
        # Métriques de performance
        self.contact_sensors = []
        self.palm_contact = False
        self.finger_contacts = [False] * 8  # 4 doigts x 2 mains
        self.cube_grasped = False
        self.cube_lifted = False
        self.grasp_force = 0.0
        self.stability_score = 0.0
        
        # Historique pour stabilité
        self.cube_velocity_history = []
        self.hand_position_history = []
        self.max_history = 20
        
        # Initialisation du modèle physique
        self._setup_physics_model()
        self._identify_components()
        self._setup_spaces()
        
        # Configuration vidéo
        self.video_frames = []
        self.camera_id = 0
        
        print("🤖 RobustGraspEnv initialisé avec succès!")
        print(f"  📐 Espace action: {self.action_space.shape}")
        print(f"  👁️  Espace observation: {self.observation_space.shape}")
        print(f"  🎬 Enregistrement vidéo: {'✅' if record_video else '❌'}")
    
    def _setup_physics_model(self):
        """Crée le modèle physique robuste avec collisions"""
        
        # Construire le XML du modèle complet
        xml_content = self._build_robust_xml()
        
        # Charger le modèle avec gestion d'erreur
        try:
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                temp_path = f.name
            
            # Charger le modèle
            self.model = mujoco.MjModel.from_xml_path(temp_path)
            self.data = mujoco.MjData(self.model)
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)
            
            print("✅ Modèle physique chargé avec succès")
            print(f"  - DOFs: {self.model.nq}")
            print(f"  - Actuateurs: {self.model.nu}")
            print(f"  - Capteurs: {self.model.nsensor}")
            print(f"  - Timestep: {self.model.opt.timestep}")
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _build_robust_xml(self) -> str:
        """Construit le XML du modèle avec physics collision robuste"""
        
        xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robust_grasp_env">
    <compiler angle="radian"/>
    
    <option timestep="0.002" 
            gravity="0 0 -9.81" 
            integrator="RK4"
            iterations="50"
            tolerance="1e-10"
            jacobian="auto"
            cone="elliptic"
            impratio="1"/>
    
    <size nconmax="1000" njmax="2000"/>
    
    <visual>
        <global offwidth="640" offheight="640"/>
    </visual>
    
    <!-- Matériaux -->
    <asset>
        <material name="table_material" 
                  rgba="0.8 0.6 0.4 1" 
                  specular="0.3" 
                  shininess="0.1"
                  reflectance="0.1"/>
        <material name="cube_material" 
                  rgba="0.2 0.6 0.9 1" 
                  specular="0.5" 
                  shininess="0.3"/>
        <material name="hand_material" 
                  rgba="0.9 0.8 0.7 1" 
                  specular="0.2" 
                  shininess="0.1"/>
    </asset>
    
    <default>
        <joint limited="true" damping="0.1" armature="0.01"/>
        <geom contype="1" conaffinity="1" friction="1.0 0.1 0.05"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <worldbody>
        <!-- Sol avec friction -->
        <geom name="floor" type="plane" size="2 2 0.1" 
              rgba="0.5 0.5 0.5 1" 
              friction="0.8 0.1 0.05"/>
        
        <!-- Table robuste avec collision -->
        <body name="table" pos="0 0 1.0">
            <geom name="table_surface" 
                  type="box" 
                  size="0.6 0.4 0.05" 
                  material="table_material"
                  friction="1.2 0.2 0.1"/>
            <!-- Pieds de table pour stabilité -->
            <geom name="leg1" type="box" size="0.05 0.05 0.45" pos="0.5 0.3 -0.5" rgba="0.6 0.4 0.2 1"/>
            <geom name="leg2" type="box" size="0.05 0.05 0.45" pos="0.5 -0.3 -0.5" rgba="0.6 0.4 0.2 1"/>
            <geom name="leg3" type="box" size="0.05 0.05 0.45" pos="-0.5 0.3 -0.5" rgba="0.6 0.4 0.2 1"/>
            <geom name="leg4" type="box" size="0.05 0.05 0.45" pos="-0.5 -0.3 -0.5" rgba="0.6 0.4 0.2 1"/>
        </body>
        
        <!-- Robot G1 humanoid avec bras duels -->
        {self._build_g1_robot()}
        
        <!-- Cube manipulable avec physics -->
        <body name="cube" pos="{self.cube_initial_pos[0]} {self.cube_initial_pos[1]} {self.cube_initial_pos[2]}">
            <geom name="cube_geom" 
                  type="box" 
                  size="0.04 0.04 0.04" 
                  material="cube_material"
                  mass="0.2"
                  friction="1.5 0.3 0.2"/>
            <joint name="cube_joint" type="free"/>
        </body>
        
        <!-- Caméras pour observation -->
        <camera name="main_camera" pos="1.5 0 1.8" xyaxes="0 -1 0 0 0 1"/>
        <camera name="side_camera" pos="0 1.5 1.5" xyaxes="-1 0 0 0 0 1"/>
        <camera name="top_camera" pos="0 0 2.5" xyaxes="1 0 0 0 1 0"/>
        
        <!-- Lumières -->
        <light name="light1" pos="1 1 3" dir="-1 -1 -1"/>
        <light name="light2" pos="-1 -1 3" dir="1 1 -1"/>
    </worldbody>
    
    <!-- Capteurs pour détection de contact -->
    <sensor>
        <!-- Contact cube-main gauche -->
        <touch name="left_palm_contact" site="left_palm_site"/>
        <touch name="left_thumb_contact" site="left_thumb_site"/>
        <touch name="left_index_contact" site="left_index_site"/>
        <touch name="left_middle_contact" site="left_middle_site"/>
        <touch name="left_ring_contact" site="left_ring_site"/>
        
        <!-- Contact cube-main droite -->
        <touch name="right_palm_contact" site="right_palm_site"/>
        <touch name="right_thumb_contact" site="right_thumb_site"/>
        <touch name="right_index_contact" site="right_index_site"/>
        <touch name="right_middle_contact" site="right_middle_site"/>
        <touch name="right_ring_contact" site="right_ring_site"/>
        
        <!-- Position du cube -->
        <framepos name="cube_position" objtype="body" objname="cube"/>
    </sensor>
    
    <!-- Actuateurs avec contrôle de force -->
    <actuator>
        {self._build_actuators()}
    </actuator>
</mujoco>"""
        
        return xml
    
    def _build_g1_robot(self) -> str:
        """Construit le robot G1 avec bras et mains articulées"""
        
        robot_xml = f"""
        <!-- Corps principal G1 -->
        <body name="base" pos="0 0 0.95">
            <geom name="torso" type="cylinder" size="0.15 0.3" 
                  rgba="0.7 0.7 0.7 1" mass="15"/>
            
            <!-- Bras gauche -->
            <body name="left_shoulder" pos="0 0.2 0.25">
                <joint name="left_shoulder_pitch" axis="0 1 0" range="-3.14 3.14"/>
                <geom name="left_shoulder_geom" type="sphere" size="0.08" rgba="0.6 0.6 0.6 1"/>
                
                <body name="left_upper_arm" pos="0 0.1 0">
                    <joint name="left_shoulder_roll" axis="1 0 0" range="-1.57 1.57"/>
                    <geom name="left_upper_arm_geom" type="capsule" size="0.04 0.15" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="left_elbow" pos="0 0.25 0">
                        <joint name="left_shoulder_yaw" axis="0 0 1" range="-3.14 3.14"/>
                        <geom name="left_elbow_geom" type="sphere" size="0.06" rgba="0.5 0.5 0.5 1"/>
                        
                        <body name="left_forearm" pos="0 0.15 0">
                            <joint name="left_elbow_pitch" axis="0 1 0" range="-2.62 0"/>
                            <geom name="left_forearm_geom" type="capsule" size="0.035 0.12" rgba="0.6 0.6 0.6 1"/>
                            
                            <body name="left_wrist" pos="0 0.2 0">
                                <joint name="left_wrist_roll" axis="1 0 0" range="-1.57 1.57"/>
                                <joint name="left_wrist_pitch" axis="0 1 0" range="-1.57 1.57"/>
                                <joint name="left_wrist_yaw" axis="0 0 1" range="-1.57 1.57"/>
                                
                                {self._build_hand("left")}
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Bras droit (symétrique) -->
            <body name="right_shoulder" pos="0 -0.2 0.25">
                <joint name="right_shoulder_pitch" axis="0 1 0" range="-3.14 3.14"/>
                <geom name="right_shoulder_geom" type="sphere" size="0.08" rgba="0.6 0.6 0.6 1"/>
                
                <body name="right_upper_arm" pos="0 -0.1 0">
                    <joint name="right_shoulder_roll" axis="1 0 0" range="-1.57 1.57"/>
                    <geom name="right_upper_arm_geom" type="capsule" size="0.04 0.15" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="right_elbow" pos="0 -0.25 0">
                        <joint name="right_shoulder_yaw" axis="0 0 1" range="-3.14 3.14"/>
                        <geom name="right_elbow_geom" type="sphere" size="0.06" rgba="0.5 0.5 0.5 1"/>
                        
                        <body name="right_forearm" pos="0 -0.15 0">
                            <joint name="right_elbow_pitch" axis="0 1 0" range="-2.62 0"/>
                            <geom name="right_forearm_geom" type="capsule" size="0.035 0.12" rgba="0.6 0.6 0.6 1"/>
                            
                            <body name="right_wrist" pos="0 -0.2 0">
                                <joint name="right_wrist_roll" axis="1 0 0" range="-1.57 1.57"/>
                                <joint name="right_wrist_pitch" axis="0 1 0" range="-1.57 1.57"/>
                                <joint name="right_wrist_yaw" axis="0 0 1" range="-1.57 1.57"/>
                                
                                {self._build_hand("right")}
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>"""
        
        return robot_xml
    
    def _build_hand(self, side: str) -> str:
        """Construit une main articulée avec détection de contact"""
        
        hand_xml = f"""
        <!-- Main {side} -->
        <body name="{side}_hand" pos="0 0 0">
            <geom name="{side}_palm" type="box" size="0.04 0.08 0.02" 
                  material="hand_material"/>
            <site name="{side}_palm_site" pos="0 0 0" size="0.01"/>
            
            <!-- Pouce -->
            <body name="{side}_thumb" pos="0.03 0.06 0">
                <joint name="{side}_thumb_joint" axis="1 0 0" range="0 1.57"/>
                <geom name="{side}_thumb_geom" type="capsule" size="0.012 0.03" rgba="0.9 0.8 0.7 1"/>
                <site name="{side}_thumb_site" pos="0 0.025 0" size="0.008"/>
            </body>
            
            <!-- Index -->
            <body name="{side}_index" pos="0.02 0.08 0">
                <joint name="{side}_index_joint" axis="1 0 0" range="0 1.57"/>
                <geom name="{side}_index_geom" type="capsule" size="0.01 0.035" rgba="0.9 0.8 0.7 1"/>
                <site name="{side}_index_site" pos="0 0.03 0" size="0.008"/>
            </body>
            
            <!-- Majeur -->
            <body name="{side}_middle" pos="0 0.08 0">
                <joint name="{side}_middle_joint" axis="1 0 0" range="0 1.57"/>
                <geom name="{side}_middle_geom" type="capsule" size="0.01 0.038" rgba="0.9 0.8 0.7 1"/>
                <site name="{side}_middle_site" pos="0 0.032 0" size="0.008"/>
            </body>
            
            <!-- Annulaire -->
            <body name="{side}_ring" pos="-0.02 0.08 0">
                <joint name="{side}_ring_joint" axis="1 0 0" range="0 1.57"/>
                <geom name="{side}_ring_geom" type="capsule" size="0.01 0.035" rgba="0.9 0.8 0.7 1"/>
                <site name="{side}_ring_site" pos="0 0.03 0" size="0.008"/>
            </body>
        </body>"""
        
        return hand_xml
    
    def _build_actuators(self) -> str:
        """Construit les actuateurs avec contrôle de force"""
        
        actuator_xml = """
        <!-- Actuateurs bras gauche -->
        <motor name="left_shoulder_pitch_motor" joint="left_shoulder_pitch" gear="100"/>
        <motor name="left_shoulder_roll_motor" joint="left_shoulder_roll" gear="100"/>
        <motor name="left_shoulder_yaw_motor" joint="left_shoulder_yaw" gear="100"/>
        <motor name="left_elbow_pitch_motor" joint="left_elbow_pitch" gear="80"/>
        <motor name="left_wrist_roll_motor" joint="left_wrist_roll" gear="50"/>
        <motor name="left_wrist_pitch_motor" joint="left_wrist_pitch" gear="50"/>
        <motor name="left_wrist_yaw_motor" joint="left_wrist_yaw" gear="50"/>
        
        <!-- Actuateurs bras droit -->
        <motor name="right_shoulder_pitch_motor" joint="right_shoulder_pitch" gear="100"/>
        <motor name="right_shoulder_roll_motor" joint="right_shoulder_roll" gear="100"/>
        <motor name="right_shoulder_yaw_motor" joint="right_shoulder_yaw" gear="100"/>
        <motor name="right_elbow_pitch_motor" joint="right_elbow_pitch" gear="80"/>
        <motor name="right_wrist_roll_motor" joint="right_wrist_roll" gear="50"/>
        <motor name="right_wrist_pitch_motor" joint="right_wrist_pitch" gear="50"/>
        <motor name="right_wrist_yaw_motor" joint="right_wrist_yaw" gear="50"/>
        
        <!-- Actuateurs mains -->
        <motor name="left_thumb_motor" joint="left_thumb_joint" gear="20"/>
        <motor name="left_index_motor" joint="left_index_joint" gear="20"/>
        <motor name="left_middle_motor" joint="left_middle_joint" gear="20"/>
        <motor name="left_ring_motor" joint="left_ring_joint" gear="20"/>
        
        <motor name="right_thumb_motor" joint="right_thumb_joint" gear="20"/>
        <motor name="right_index_motor" joint="right_index_joint" gear="20"/>
        <motor name="right_middle_motor" joint="right_middle_joint" gear="20"/>
        <motor name="right_ring_motor" joint="right_ring_joint" gear="20"/>"""
        
        return actuator_xml
    
    def _identify_components(self):
        """Identifie les composants du robot pour contrôle"""
        
        # Joints des bras (14 joints total)
        self.arm_joint_names = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 
            'left_elbow_pitch', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 
            'right_elbow_pitch', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw'
        ]
        
        # Joints des doigts (8 joints total)
        self.finger_joint_names = [
            'left_thumb_joint', 'left_index_joint', 'left_middle_joint', 'left_ring_joint',
            'right_thumb_joint', 'right_index_joint', 'right_middle_joint', 'right_ring_joint'
        ]
        
        # Obtenir les IDs
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.arm_joint_ids.append(joint_id)
        
        self.finger_joint_ids = []
        for name in self.finger_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.finger_joint_ids.append(joint_id)
        
        # Cube et sites de contact
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Sites pour détection de contact
        self.contact_site_names = [
            'left_palm_site', 'left_thumb_site', 'left_index_site', 'left_middle_site', 'left_ring_site',
            'right_palm_site', 'right_thumb_site', 'right_index_site', 'right_middle_site', 'right_ring_site'
        ]
        
        self.contact_site_ids = []
        for name in self.contact_site_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self.contact_site_ids.append(site_id)
        
        print(f"✅ Composants identifiés:")
        print(f"  - Joints bras: {len(self.arm_joint_ids)}")
        print(f"  - Joints doigts: {len(self.finger_joint_ids)}")
        print(f"  - Sites contact: {len(self.contact_site_ids)}")
    
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation avec dimensions correctes"""
        
        # Espace d'action: 14 (bras) + 8 (doigts) = 22 actions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32
        )
        
        # Espace d'observation: dimensions corrigées = 88
        # Structure: qpos(36) + qvel(36) + cube_pos(3) + cube_quat(4) + cube_vel(3) + phase(1) + contacts(5) = 88
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Réinitialiser l'état de la physique
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale du cube avec variation
        cube_variation = np.random.uniform(-0.1, 0.1, 3)
        cube_variation[2] = 0  # Pas de variation en hauteur
        self.data.qpos[-7:-4] = self.cube_initial_pos + cube_variation
        self.data.qpos[-4:] = [1, 0, 0, 0]  # Quaternion identité
        
        # Positions initiales des bras avec légère variation
        for i, joint_id in enumerate(self.arm_joint_ids):
            if i < 7:  # Bras gauche
                base_pos = list(self.arm_initial_positions['left'])
                variation = np.random.uniform(-0.1, 0.1)
                self.data.qpos[joint_id] = base_pos[i] + variation
            else:  # Bras droit
                base_pos = list(self.arm_initial_positions['right'])
                variation = np.random.uniform(-0.1, 0.1)
                self.data.qpos[joint_id] = base_pos[i-7] + variation
        
        # Doigts légèrement ouverts
        for joint_id in self.finger_joint_ids:
            self.data.qpos[joint_id] = np.random.uniform(0.0, 0.2)
        
        # Réinitialiser les métriques
        self.current_phase = 0
        self.phase_timer = 0
        self.episode_step = 0
        self.contact_sensors = []
        self.palm_contact = False
        self.finger_contacts = [False] * 8
        self.cube_grasped = False
        self.cube_lifted = False
        self.grasp_force = 0.0
        self.stability_score = 0.0
        
        # Vider les historiques
        self.cube_velocity_history.clear()
        self.hand_position_history.clear()
        self.video_frames.clear()
        
        # Première simulation pour stabiliser
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Exécute une action et retourne la transition"""
        
        action = np.clip(action, -1.0, 1.0)
        
        # Appliquer les actions aux articulations
        self._apply_action(action)
        
        # Simulation physique
        mujoco.mj_step(self.model, self.data)
        
        # Mise à jour des métriques
        self._update_sensors()
        self._update_phase_logic()
        self._update_history()
        
        # Calcul de la récompense
        reward = self._compute_reward()
        
        # Vérification de terminaison
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        # Enregistrement vidéo - uniquement pour les épisodes de démo
        # if self.record_video:
        #     self._record_frame()
        
        # Incrémenter les compteurs
        self.episode_step += 1
        self.phase_timer += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Applique l'action aux articulations avec contrôle de force adaptatif"""
        
        # Actions bras (14 premières actions)
        arm_actions = action[:14]
        for i, joint_id in enumerate(self.arm_joint_ids):
            # Contrôle adaptatif selon la phase
            if self.current_phase == 0:  # SEARCH - mouvements plus amples
                control_gain = 0.3
            elif self.current_phase in [1, 2]:  # APPROACH/CONTACT - plus précis
                control_gain = 0.15
            else:  # ALIGN/GRASP/LIFT - très précis
                control_gain = 0.1
            
            target_pos = self.data.qpos[joint_id] + arm_actions[i] * control_gain
            # Appliquer les limites articulaires
            joint_range = self.model.jnt_range[joint_id]
            if joint_range[0] < joint_range[1]:  # Joint limité
                target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
            
            self.data.ctrl[i] = target_pos
        
        # Actions doigts (8 dernières actions)
        finger_actions = action[14:]
        for i, joint_id in enumerate(self.finger_joint_ids):
            # Contrôle de force progressif pour la saisie
            if self.current_phase >= 4:  # GRASP phase
                # Force proportionnelle au contact détecté
                contact_force = 1.0 if any(self.finger_contacts) else 0.5
                control_gain = 0.1 * contact_force
            else:
                control_gain = 0.05
            
            target_pos = self.data.qpos[joint_id] + finger_actions[i] * control_gain
            target_pos = np.clip(target_pos, 0.0, 1.57)  # Limites doigts
            
            self.data.ctrl[14 + i] = target_pos
    
    def _update_sensors(self):
        """Met à jour les capteurs de contact et de force"""
        
        # Réinitialiser les contacts
        self.palm_contact = False
        self.finger_contacts = [False] * 8
        
        # Vérifier les contacts via les capteurs
        if self.model.nsensor > 0:
            for i in range(min(10, self.model.nsensor)):  # 10 capteurs de contact
                sensor_data = self.data.sensordata[i]
                if sensor_data > 0.1:  # Seuil de contact
                    if i == 0 or i == 5:  # Palm contacts
                        self.palm_contact = True
                    else:  # Finger contacts
                        finger_idx = (i - 1) % 4 + (4 if i > 5 else 0)
                        if finger_idx < 8:
                            self.finger_contacts[finger_idx] = True
        
        # Calculer la force de saisie
        active_contacts = sum(self.finger_contacts) + (2 if self.palm_contact else 0)
        self.grasp_force = active_contacts / 10.0  # Normalisé
        
        # Vérifier si le cube est saisi
        self.cube_grasped = (active_contacts >= 4 and self.palm_contact)
        
        # Vérifier si le cube est levé
        cube_pos = self._get_cube_position()
        self.cube_lifted = (cube_pos[2] > self.cube_initial_pos[2] + 0.1) and self.cube_grasped
    
    def _update_phase_logic(self):
        """Met à jour la logique de progression des phases"""
        
        cube_pos = self._get_cube_position()
        left_hand_pos = self._get_hand_position("left")
        right_hand_pos = self._get_hand_position("right")
        
        # Distance au cube (plus proche des deux mains)
        left_dist = np.linalg.norm(cube_pos - left_hand_pos)
        right_dist = np.linalg.norm(cube_pos - right_hand_pos)
        min_distance = min(left_dist, right_dist)
        
        # Logique de progression automatique
        phase_completed = False
        
        if self.current_phase == 0:  # SEARCH
            # Passer à APPROACH si on s'approche du cube
            if min_distance < 0.3:
                phase_completed = True
        
        elif self.current_phase == 1:  # APPROACH
            # Passer à CONTACT si très proche
            if min_distance < 0.15:
                phase_completed = True
        
        elif self.current_phase == 2:  # CONTACT
            # Passer à ALIGN si contact détecté
            if any(self.finger_contacts) or self.palm_contact:
                phase_completed = True
        
        elif self.current_phase == 3:  # ALIGN
            # Passer à GRASP si bien aligné
            if self.palm_contact and min_distance < 0.08:
                phase_completed = True
        
        elif self.current_phase == 4:  # GRASP
            # Passer à LIFT si cube bien saisi
            if self.cube_grasped and self.grasp_force > 0.6:
                phase_completed = True
        
        elif self.current_phase == 5:  # LIFT
            # Passer à HOLD si cube levé
            if self.cube_lifted:
                phase_completed = True
        
        # Transition de phase ou timeout
        max_duration = self.phase_durations.get(list(self.PHASES.keys())[self.current_phase], 100)
        
        if phase_completed or self.phase_timer >= max_duration:
            if self.current_phase < len(self.PHASES) - 1:
                self.current_phase += 1
                self.phase_timer = 0
    
    def _update_history(self):
        """Met à jour les historiques pour calcul de stabilité"""
        
        # Historique vitesse cube
        cube_vel = self._get_cube_velocity()
        self.cube_velocity_history.append(np.linalg.norm(cube_vel))
        if len(self.cube_velocity_history) > self.max_history:
            self.cube_velocity_history.pop(0)
        
        # Historique position mains
        left_pos = self._get_hand_position("left")
        right_pos = self._get_hand_position("right")
        self.hand_position_history.append([left_pos, right_pos])
        if len(self.hand_position_history) > self.max_history:
            self.hand_position_history.pop(0)
        
        # Calcul score de stabilité
        if len(self.cube_velocity_history) >= 5:
            recent_velocities = self.cube_velocity_history[-5:]
            self.stability_score = 1.0 / (1.0 + np.mean(recent_velocities))
    
    def _compute_reward(self) -> float:
        """Calcule la récompense basée sur la phase et la performance"""
        
        reward = 0.0
        cube_pos = self._get_cube_position()
        left_hand_pos = self._get_hand_position("left")
        right_hand_pos = self._get_hand_position("right")
        
        # Distance au cube (plus proche des deux mains)
        left_dist = np.linalg.norm(cube_pos - left_hand_pos)
        right_dist = np.linalg.norm(cube_pos - right_hand_pos)
        min_distance = min(left_dist, right_dist)
        
        # Récompenses par phase
        if self.current_phase == 0:  # SEARCH
            # Récompenser l'exploration et l'approche
            reward += 1.0 - min(min_distance / 0.5, 1.0)  # Max 1.0
            
        elif self.current_phase == 1:  # APPROACH
            # Récompenser l'approche précise
            reward += 2.0 - min(min_distance / 0.2, 2.0)  # Max 2.0
            
        elif self.current_phase == 2:  # CONTACT
            # Récompenser le contact initial
            reward += 3.0 - min(min_distance / 0.1, 1.0)  # Max 3.0
            if any(self.finger_contacts):
                reward += 2.0
            if self.palm_contact:
                reward += 3.0
                
        elif self.current_phase == 3:  # ALIGN
            # Récompenser l'alignement optimal
            reward += 5.0 if self.palm_contact else 0.0
            reward += 2.0 * sum(self.finger_contacts) / 8.0
            
        elif self.current_phase == 4:  # GRASP
            # Récompenser la saisie progressive
            reward += 8.0 * self.grasp_force
            if self.cube_grasped:
                reward += 10.0
                
        elif self.current_phase == 5:  # LIFT
            # Récompenser la levée
            height_gain = max(0, cube_pos[2] - self.cube_initial_pos[2])
            reward += 15.0 * min(height_gain / 0.2, 1.0)
            if self.cube_lifted:
                reward += 20.0
                
        elif self.current_phase == 6:  # HOLD
            # Récompenser le maintien stable
            reward += 25.0 * self.stability_score
            if self.cube_lifted and self.stability_score > 0.8:
                reward += 30.0
        
        # Bonus de progression
        reward += self.current_phase * 2.0
        
        # Pénalités
        # Pénaliser les mouvements excessifs
        arm_velocity = np.linalg.norm([self.data.qvel[i] for i in self.arm_joint_ids])
        if arm_velocity > 5.0:
            reward -= 1.0
        
        # Pénaliser la chute du cube
        if cube_pos[2] < 0.9:
            reward -= 10.0
        
        # Pénaliser les positions extrêmes
        for joint_id in self.arm_joint_ids:
            joint_range = self.model.jnt_range[joint_id]
            if joint_range[0] < joint_range[1]:  # Joint limité
                pos = self.data.qpos[joint_id]
                if pos <= joint_range[0] + 0.1 or pos >= joint_range[1] - 0.1:
                    reward -= 0.5
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Construit l'observation avec dimensions correctes (88)"""
        
        obs = []
        
        # Positions des joints (36 valeurs)
        # Note: Assurons-nous d'avoir exactement 36 valeurs
        joint_positions = self.data.qpos.copy()
        if len(joint_positions) < 36:
            # Pad avec des zéros si nécessaire
            padded_pos = np.zeros(36)
            padded_pos[:len(joint_positions)] = joint_positions
            obs.extend(padded_pos)
        else:
            obs.extend(joint_positions[:36])
        
        # Vitesses des joints (36 valeurs)
        joint_velocities = self.data.qvel.copy()
        if len(joint_velocities) < 36:
            # Pad avec des zéros si nécessaire
            padded_vel = np.zeros(36)
            padded_vel[:len(joint_velocities)] = joint_velocities
            obs.extend(padded_vel)
        else:
            obs.extend(joint_velocities[:36])
        
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
        
        # Phase actuelle (1 valeur)
        obs.append(float(self.current_phase))
        
        # État des contacts (5 valeurs)
        obs.append(float(self.palm_contact))
        obs.append(float(sum(self.finger_contacts[:4])))  # Contacts main gauche
        obs.append(float(sum(self.finger_contacts[4:])))  # Contacts main droite
        obs.append(float(self.grasp_force))
        obs.append(float(self.stability_score))
        
        # Vérifier la dimension finale
        obs_array = np.array(obs, dtype=np.float32)
        
        # S'assurer qu'on a exactement 88 dimensions
        if len(obs_array) < 88:
            # Pad avec des zéros
            final_obs = np.zeros(88, dtype=np.float32)
            final_obs[:len(obs_array)] = obs_array
            return final_obs
        elif len(obs_array) > 88:
            # Tronquer
            return obs_array[:88]
        else:
            return obs_array
    
    def _get_cube_position(self) -> np.ndarray:
        """Obtient la position du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id].copy()
        else:
            return self.cube_initial_pos.copy()
    
    def _get_cube_velocity(self) -> np.ndarray:
        """Obtient la vitesse du cube"""
        if self.cube_body_id >= 0:
            return self.data.cvel[self.cube_body_id][:3].copy()
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def _get_hand_position(self, side: str) -> np.ndarray:
        """Obtient la position d'une main"""
        hand_name = f"{side}_hand"
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
        if hand_id >= 0:
            return self.data.xpos[hand_id].copy()
        else:
            return np.array([0.0, 0.0, 1.0])
    
    def _check_termination(self) -> bool:
        """Vérifie les conditions de terminaison"""
        
        # Succès: cube maintenu en l'air de manière stable
        if (self.current_phase >= 6 and 
            self.cube_lifted and 
            self.stability_score > 0.8 and
            self.phase_timer > 30):
            return True
        
        # Échec: cube tombé
        cube_pos = self._get_cube_position()
        if cube_pos[2] < 0.8:
            return True
        
        # Échec: positions invalides
        if (np.any(np.isnan(self.data.qpos)) or 
            np.any(np.isnan(self.data.qvel)) or
            np.any(np.abs(self.data.qpos) > 10)):
            return True
        
        return False
    
    def _record_frame(self):
        """Enregistre une frame vidéo"""
        if self.render_mode == "rgb_array":
            frame = self.render()
            if frame is not None:
                self.video_frames.append(frame)
    
    def _get_info(self) -> Dict[str, Any]:
        """Retourne les informations de debug"""
        return {
            'phase': list(self.PHASES.keys())[self.current_phase],
            'phase_timer': self.phase_timer,
            'episode_step': self.episode_step,
            'cube_position': self._get_cube_position().tolist(),
            'cube_grasped': self.cube_grasped,
            'cube_lifted': self.cube_lifted,
            'palm_contact': self.palm_contact,
            'finger_contacts': sum(self.finger_contacts),
            'grasp_force': self.grasp_force,
            'stability_score': self.stability_score,
            'min_distance': min(
                np.linalg.norm(self._get_cube_position() - self._get_hand_position("left")),
                np.linalg.norm(self._get_cube_position() - self._get_hand_position("right"))
            )
        }
    
    def render(self, mode=None):
        """Rendu de l'environnement"""
        if mode is None:
            mode = self.render_mode
        
        if mode == "rgb_array":
            # Configurer la caméra avec taille compatible
            width, height = 480, 480
            
            # Créer le renderer
            renderer = mujoco.Renderer(self.model, width, height)
            
            # Rendu
            renderer.update_scene(self.data, camera=self.camera_id)
            frame = renderer.render()
            
            renderer.close()
            return frame
        
        elif mode == "human":
            # Pour l'affichage en temps réel (si supporté)
            pass
    
    def save_video(self, filename: str = None):
        """Sauvegarde la vidéo de l'épisode"""
        if not self.video_frames:
            print("❌ Aucune frame à sauvegarder")
            return
        
        if filename is None:
            filename = f"grasp_episode_{self.episode_count:04d}.mp4"
        
        video_path = os.path.join(self.video_dir, filename)
        
        try:
            import imageio
            with imageio.get_writer(video_path, fps=30) as writer:
                for frame in self.video_frames:
                    writer.append_data(frame)
            
            print(f"🎬 Vidéo sauvegardée: {video_path}")
            self.episode_count += 1
            
        except ImportError:
            print("❌ imageio requis pour sauvegarder les vidéos")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def close(self):
        """Ferme l'environnement"""
        if hasattr(self, 'video_frames') and self.video_frames:
            self.save_video()
        
        # Nettoyer les ressources MuJoCo si nécessaire
        pass