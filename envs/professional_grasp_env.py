#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT DE GRASPING PROFESSIONNEL
=========================================

Environnement avec stratégie progressive et système de récompenses sophistiqué.
Approche professionnelle pour aboutir aux résultats attendus, pas juste "fonctionner".
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os
import json
import tempfile
import warnings
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import time

warnings.filterwarnings("ignore")

class TrainingStage(Enum):
    """Étapes d'entraînement progressif"""
    STAGE_1_APPROACH = "approach"           # Apprendre à s'approcher
    STAGE_2_CONTACT = "contact"             # Apprendre le contact
    STAGE_3_GRASP = "grasp"                 # Apprendre à saisir
    STAGE_4_LIFT = "lift"                   # Apprendre à soulever
    STAGE_5_MASTERY = "mastery"             # Maîtrise complète

@dataclass
class StageConfig:
    """Configuration d'une étape d'entraînement"""
    name: str
    description: str
    max_episode_steps: int
    success_threshold: float
    episodes_for_advancement: int
    reward_weights: Dict[str, float]
    termination_conditions: Dict[str, float]
    assistance_level: float  # 0.0 = aucune aide, 1.0 = aide maximale

class ProfessionalGraspEnv(gym.Env):
    """
    🎯 Environnement de grasping professionnel avec curriculum adaptatif
    
    Stratégie progressive :
    1. D'abord apprendre à s'approcher du cube
    2. Puis apprendre le contact précis
    3. Ensuite maîtriser la saisie
    4. Enfin apprendre à soulever
    5. Perfectionnement global
    """
    
    def __init__(self, 
                 model_path: str = None,
                 render_mode: str = "rgb_array",
                 stage: TrainingStage = TrainingStage.STAGE_1_APPROACH,
                 auto_progression: bool = True):
        
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.model_path = model_path or self._create_professional_model()
        self.current_stage = stage
        self.auto_progression = auto_progression
        
        # Logger professionnel
        self._setup_logging()
        
        # Configuration des étapes
        self._setup_curriculum_stages()
        
        # Chargement du modèle
        self._load_model()
        
        # Configuration des composants
        self._setup_robot_components()
        self._setup_spaces()
        
        # Variables d'état professionnelles
        self._initialize_professional_state()
        
        self.logger.info(f"🎯 Environnement professionnel initialisé - Étape: {self.current_stage.value}")
    
    def _setup_logging(self):
        """Configuration du logger professionnel"""
        self.logger = logging.getLogger("ProfessionalGrasp")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_curriculum_stages(self):
        """Configuration des étapes du curriculum professionnel"""
        self.stage_configs = {
            TrainingStage.STAGE_1_APPROACH: StageConfig(
                name="Approche",
                description="Apprendre à s'approcher du cube efficacement",
                max_episode_steps=200,
                success_threshold=15.0,  # Reward moyen pour passer à l'étape suivante
                episodes_for_advancement=50,
                reward_weights={
                    'approach': 1.0,
                    'contact': 0.1,
                    'grasp': 0.0,
                    'lift': 0.0,
                    'stability': 0.5
                },
                termination_conditions={
                    'max_distance': 0.5,
                    'min_cube_height': 0.4,
                    'max_cube_height': 0.8
                },
                assistance_level=0.8  # Aide importante pour apprendre
            ),
            
            TrainingStage.STAGE_2_CONTACT: StageConfig(
                name="Contact",
                description="Maîtriser le contact précis avec le cube",
                max_episode_steps=300,
                success_threshold=25.0,
                episodes_for_advancement=50,
                reward_weights={
                    'approach': 0.5,
                    'contact': 1.0,
                    'grasp': 0.3,
                    'lift': 0.0,
                    'stability': 0.7
                },
                termination_conditions={
                    'max_distance': 0.4,
                    'min_cube_height': 0.35,
                    'max_cube_height': 0.9
                },
                assistance_level=0.6
            ),
            
            TrainingStage.STAGE_3_GRASP: StageConfig(
                name="Saisie",
                description="Développer une saisie stable et efficace",
                max_episode_steps=400,
                success_threshold=40.0,
                episodes_for_advancement=75,
                reward_weights={
                    'approach': 0.2,
                    'contact': 0.7,
                    'grasp': 1.0,
                    'lift': 0.2,
                    'stability': 0.8
                },
                termination_conditions={
                    'max_distance': 0.6,
                    'min_cube_height': 0.3,
                    'max_cube_height': 1.0
                },
                assistance_level=0.4
            ),
            
            TrainingStage.STAGE_4_LIFT: StageConfig(
                name="Levage",
                description="Apprendre à soulever et maintenir le cube",
                max_episode_steps=500,
                success_threshold=60.0,
                episodes_for_advancement=100,
                reward_weights={
                    'approach': 0.1,
                    'contact': 0.5,
                    'grasp': 0.8,
                    'lift': 1.0,
                    'stability': 1.0
                },
                termination_conditions={
                    'max_distance': 0.8,
                    'min_cube_height': 0.2,
                    'max_cube_height': 1.2
                },
                assistance_level=0.2
            ),
            
            TrainingStage.STAGE_5_MASTERY: StageConfig(
                name="Maîtrise",
                description="Perfectionnement et robustesse complète",
                max_episode_steps=500,
                success_threshold=80.0,
                episodes_for_advancement=150,
                reward_weights={
                    'approach': 0.3,
                    'contact': 0.7,
                    'grasp': 1.0,
                    'lift': 1.0,
                    'stability': 1.0
                },
                termination_conditions={
                    'max_distance': 1.0,
                    'min_cube_height': 0.1,
                    'max_cube_height': 1.5
                },
                assistance_level=0.0  # Aucune aide
            )
        }
    
    def _create_professional_model(self) -> str:
        """Créer un modèle professionnel optimisé"""
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="professional_grasp">
    <compiler angle="radian" meshdir="." texturedir="."/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" solver="PGS" iterations="50"/>
    
    <default>
        <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.1 0.05"/>
        <joint damping="0.1" stiffness="0"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <asset>
        <material name="table_mat" rgba="0.8 0.6 0.4 1" specular="0.3"/>
        <material name="cube_mat" rgba="0.2 0.6 0.8 1" specular="0.5"/>
        <material name="hand_mat" rgba="0.9 0.7 0.5 1" specular="0.3"/>
    </asset>
    
    <worldbody>
        <!-- Environment lighting -->
        <light name="top_light" pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light name="side_light" pos="1 1 1" dir="-1 -1 -1" diffuse="0.3 0.3 0.3"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 0.4">
            <geom type="box" size="0.6 0.6 0.05" material="table_mat" mass="50"/>
        </body>
        
        <!-- Robot base -->
        <body name="robot_base" pos="0 0 0.5">
            <!-- Shoulder -->
            <body name="shoulder" pos="0 -0.15 0.2">
                <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                <joint name="shoulder_tilt" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04 0.08" rgba="0.7 0.7 0.7 1"/>
                
                <!-- Upper arm -->
                <body name="upper_arm" pos="0 0 -0.15">
                    <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.5"/>
                    <geom type="capsule" size="0.03 0.1" rgba="0.6 0.6 0.6 1"/>
                    
                    <!-- Forearm -->
                    <body name="forearm" pos="0 0 -0.15">
                        <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                        <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom type="capsule" size="0.025 0.08" rgba="0.5 0.5 0.5 1"/>
                        
                        <!-- Hand -->
                        <body name="hand" pos="0 0 -0.1">
                            <geom type="box" size="0.03 0.04 0.02" material="hand_mat"/>
                            
                            <!-- Thumb -->
                            <body name="thumb_base" pos="0.02 0.03 0">
                                <joint name="thumb_base_joint" type="hinge" axis="1 0 0" range="-0.5 1.2"/>
                                <geom name="thumb_base_geom" type="capsule" size="0.01 0.02" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="thumb_tip" pos="0.015 0 0">
                                    <joint name="thumb_tip_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="thumb_tip_geom" type="capsule" size="0.008 0.015" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                            
                            <!-- Index finger -->
                            <body name="index_base" pos="0.04 0.01 0">
                                <joint name="index_base_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="index_base_geom" type="capsule" size="0.01 0.025" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="index_tip" pos="0.02 0 0">
                                    <joint name="index_tip_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="index_tip_geom" type="capsule" size="0.008 0.02" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                            
                            <!-- Middle finger -->
                            <body name="middle_base" pos="0.04 -0.01 0">
                                <joint name="middle_base_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="middle_base_geom" type="capsule" size="0.01 0.025" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="middle_tip" pos="0.02 0 0">
                                    <joint name="middle_tip_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="middle_tip_geom" type="capsule" size="0.008 0.02" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                            
                            <!-- Ring finger (passive) -->
                            <body name="ring_base" pos="0.04 -0.03 0">
                                <joint name="ring_base_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                <geom name="ring_base_geom" type="capsule" size="0.01 0.02" rgba="0.9 0.7 0.5 1"/>
                                
                                <body name="ring_tip" pos="0.015 0 0">
                                    <joint name="ring_tip_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                                    <geom name="ring_tip_geom" type="capsule" size="0.008 0.015" rgba="0.9 0.7 0.5 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Target cube -->
        <body name="cube" pos="0.2 0 0.5">
            <joint name="cube_joint" type="free"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" 
                  material="cube_mat" priority="1"/>
        </body>
        
        <!-- Camera views -->
        <camera name="main_view" pos="0.8 0.5 0.8" xyaxes="1 0 0 0 1 1"/>
        <camera name="side_view" pos="0 0.8 0.6" xyaxes="1 0 0 0 0 1"/>
    </worldbody>
    
    <actuator>
        <!-- Arm actuators -->
        <motor name="shoulder_pan_motor" joint="shoulder_pan" gear="100"/>
        <motor name="shoulder_tilt_motor" joint="shoulder_tilt" gear="100"/>
        <motor name="elbow_motor" joint="elbow" gear="80"/>
        <motor name="wrist_roll_motor" joint="wrist_roll" gear="40"/>
        <motor name="wrist_pitch_motor" joint="wrist_pitch" gear="40"/>
        
        <!-- Hand actuators -->
        <motor name="thumb_base_motor" joint="thumb_base_joint" gear="20"/>
        <motor name="thumb_tip_motor" joint="thumb_tip_joint" gear="15"/>
        <motor name="index_base_motor" joint="index_base_joint" gear="20"/>
        <motor name="index_tip_motor" joint="index_tip_joint" gear="15"/>
        <motor name="middle_base_motor" joint="middle_base_joint" gear="20"/>
        <motor name="middle_tip_motor" joint="middle_tip_joint" gear="15"/>
        <motor name="ring_base_motor" joint="ring_base_joint" gear="15"/>
        <motor name="ring_tip_motor" joint="ring_tip_joint" gear="10"/>
    </actuator>
    
    <sensor>
        <!-- Position sensors -->
        <framepos name="hand_pos" objtype="body" objname="hand"/>
        <framepos name="cube_pos" objtype="body" objname="cube"/>
        
        <!-- Contact sensors -->
        <touch name="thumb_contact" site="thumb_tip"/>
        <touch name="index_contact" site="index_tip"/>
        <touch name="middle_contact" site="middle_tip"/>
    </sensor>
</mujoco>'''
        
        # Sauvegarder dans un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(model_xml)
        temp_file.flush()
        temp_file.close()
        
        self.temp_model_file = temp_file.name
        return temp_file.name
    
    def _load_model(self):
        """Chargement professionnel du modèle"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Configuration physique optimisée
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-10
            
            # Renderer optionnel
            self.renderer = None
            try:
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            except Exception as e:
                self.logger.warning(f"Renderer non disponible (mode headless): {e}")
            
            self.logger.info(f"✅ Modèle professionnel chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _setup_robot_components(self):
        """Identification professionnelle des composants"""
        # Identifier tous les actuateurs
        self.actuator_names = []
        self.actuator_ids = []
        
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_names.append(name)
                self.actuator_ids.append(i)
        
        self.actuator_ids = np.array(self.actuator_ids, dtype=np.int32)
        
        # Séparer bras et main
        self.arm_actuator_ids = []
        self.hand_actuator_ids = []
        
        for i, name in enumerate(self.actuator_names):
            if any(joint in name for joint in ['shoulder', 'elbow', 'wrist']):
                self.arm_actuator_ids.append(self.actuator_ids[i])
            else:
                self.hand_actuator_ids.append(self.actuator_ids[i])
        
        self.arm_actuator_ids = np.array(self.arm_actuator_ids, dtype=np.int32)
        self.hand_actuator_ids = np.array(self.hand_actuator_ids, dtype=np.int32)
        
        # IDs des corps importants
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        self.logger.info(f"✅ Composants identifiés: {len(self.arm_actuator_ids)} bras, {len(self.hand_actuator_ids)} main")
    
    def _setup_spaces(self):
        """Configuration professionnelle des espaces"""
        # Espace d'action: tous les actuateurs
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(len(self.actuator_ids),),
            dtype=np.float32
        )
        
        # Espace d'observation sophistiqué
        # - État du robot (positions + vitesses)
        # - Position et orientation du cube
        # - Position et orientation de la main
        # - Distances et angles relatifs
        # - État des contacts
        # - Informations de curriculum
        obs_dim = (
            self.model.nq + self.model.nv +  # État robot
            7 +  # Position + orientation cube
            7 +  # Position + orientation main
            4 +  # Distances et angles
            6 +  # Contacts et forces
            4    # Informations curriculum
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.logger.info(f"✅ Espaces configurés: Action {self.action_space.shape}, Obs {self.observation_space.shape}")
    
    def _initialize_professional_state(self):
        """Initialisation professionnelle des variables d'état"""
        # Variables de base
        self.current_step = 0
        self.episode_count = 0
        self.cube_initial_pos = None
        self.hand_initial_pos = None
        
        # Variables de curriculum
        self.stage_episodes = 0
        self.stage_rewards = []
        self.stage_successes = 0
        
        # Variables de performance
        self.best_distance = float('inf')
        self.contact_history = []
        self.grasp_quality_history = []
        self.success_rate_window = []
        
        # Variables de monitoring professionnel
        self.metrics = {
            'total_episodes': 0,
            'stage_progression': [],
            'performance_history': [],
            'curriculum_transitions': []
        }
    
    def get_current_stage_config(self) -> StageConfig:
        """Obtenir la configuration de l'étape actuelle"""
        return self.stage_configs[self.current_stage]
    
    def advance_to_next_stage(self) -> bool:
        """Progression vers l'étape suivante"""
        current_stages = list(TrainingStage)
        current_index = current_stages.index(self.current_stage)
        
        if current_index < len(current_stages) - 1:
            old_stage = self.current_stage
            self.current_stage = current_stages[current_index + 1]
            
            # Reset des statistiques de l'étape
            self.stage_episodes = 0
            self.stage_rewards = []
            self.stage_successes = 0
            
            # Log de la transition
            transition_info = {
                'from_stage': old_stage.value,
                'to_stage': self.current_stage.value,
                'episode': self.episode_count,
                'timestamp': time.time()
            }
            self.metrics['curriculum_transitions'].append(transition_info)
            
            self.logger.info(f"🎓 PROGRESSION CURRICULUM: {old_stage.value} → {self.current_stage.value}")
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset professionnel avec gestion du curriculum"""
        super().reset(seed=seed)
        
        # Reset physique
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.episode_count += 1
        self.stage_episodes += 1
        
        # Configuration selon l'étape actuelle
        stage_config = self.get_current_stage_config()
        
        # Position du cube selon l'étape
        self._reset_cube_position(stage_config)
        
        # Position initiale du robot
        self._reset_robot_position(stage_config)
        
        # Stabilisation
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Initialiser les variables d'épisode
        self.best_distance = float('inf')
        self.contact_history = []
        self.grasp_quality_history = []
        
        return self._get_professional_observation(), self._get_professional_info()
    
    def _reset_cube_position(self, stage_config: StageConfig):
        """Positionnement du cube selon l'étape"""
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        
        if cube_joint_id >= 0:
            cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
            
            # Position selon l'étape
            if self.current_stage == TrainingStage.STAGE_1_APPROACH:
                # Proche pour apprendre l'approche
                x = np.random.uniform(0.15, 0.25)
                y = np.random.uniform(-0.05, 0.05)
                z = 0.5
            elif self.current_stage == TrainingStage.STAGE_2_CONTACT:
                # Positions variées pour le contact
                x = np.random.uniform(0.12, 0.28)
                y = np.random.uniform(-0.08, 0.08)
                z = np.random.uniform(0.48, 0.52)
            else:
                # Positions plus challenging
                x = np.random.uniform(0.1, 0.3)
                y = np.random.uniform(-0.1, 0.1)
                z = np.random.uniform(0.46, 0.54)
            
            cube_pos = np.array([x, y, z])
            cube_quat = np.array([1, 0, 0, 0])
            
            self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = cube_pos
            self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = cube_quat
            
            self.cube_initial_pos = cube_pos.copy()
    
    def _reset_robot_position(self, stage_config: StageConfig):
        """Position initiale du robot selon l'étape"""
        # Position de base légèrement aléatoire
        if self.current_stage == TrainingStage.STAGE_1_APPROACH:
            # Position éloignée pour apprendre l'approche
            self.data.qpos[0] = np.random.uniform(-0.3, 0.3)  # shoulder_pan
            self.data.qpos[1] = np.random.uniform(-0.5, 0.5)  # shoulder_tilt
            self.data.qpos[2] = np.random.uniform(0.5, 1.5)   # elbow
        else:
            # Position plus proche pour les étapes avancées
            self.data.qpos[0] = np.random.uniform(-0.2, 0.2)
            self.data.qpos[1] = np.random.uniform(-0.3, 0.3)
            self.data.qpos[2] = np.random.uniform(0.8, 1.2)
        
        # Main ouverte initialement
        for i in range(5, self.model.nq):  # Doigts
            self.data.qpos[i] = 0.0
    
    def step(self, action):
        """Step professionnel avec calcul sophistiqué des récompenses"""
        # Validation des actions
        action = self._validate_and_scale_actions(action)
        
        # Application des actions
        self._apply_professional_actions(action)
        
        # Simulation physique
        mujoco.mj_step(self.model, self.data)
        
        # Calcul de l'observation
        observation = self._get_professional_observation()
        
        # Calcul sophistiqué des récompenses
        reward = self._calculate_sophisticated_reward()
        
        # Vérification des conditions de terminaison
        terminated = self._check_professional_termination()
        
        # Mise à jour des métriques
        self._update_professional_metrics(reward)
        
        self.current_step += 1
        stage_config = self.get_current_stage_config()
        truncated = self.current_step >= stage_config.max_episode_steps
        
        info = self._get_professional_info()
        
        return observation, reward, terminated, truncated, info
    
    def _validate_and_scale_actions(self, action):
        """Validation et mise à l'échelle professionnelle des actions"""
        action = np.array(action, dtype=np.float32)
        
        # Gestion NaN/Inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("Action NaN/Inf détectée, remplacement par zéros")
            action = np.zeros_like(action)
        
        # Limitation selon l'étape
        stage_config = self.get_current_stage_config()
        
        # Échelles adaptatives selon l'étape
        if self.current_stage == TrainingStage.STAGE_1_APPROACH:
            arm_scale = 0.3
            hand_scale = 0.1  # Main presque fermée
        elif self.current_stage == TrainingStage.STAGE_2_CONTACT:
            arm_scale = 0.4
            hand_scale = 0.3
        else:
            arm_scale = 0.5
            hand_scale = 0.7
        
        # Application des échelles
        action[:len(self.arm_actuator_ids)] *= arm_scale
        action[len(self.arm_actuator_ids):] *= hand_scale
        
        return np.clip(action, -1.0, 1.0)
    
    def _apply_professional_actions(self, action):
        """Application professionnelle des actions avec assistance"""
        stage_config = self.get_current_stage_config()
        
        # Reset des contrôles
        self.data.ctrl[:] = 0.0
        
        # Application des actions de base
        self.data.ctrl[self.actuator_ids] = action
        
        # Assistance selon l'étape
        if stage_config.assistance_level > 0:
            self._apply_curriculum_assistance(stage_config.assistance_level)
    
    def _apply_curriculum_assistance(self, assistance_level: float):
        """Assistance adaptée selon l'étape du curriculum"""
        cube_pos = self._get_cube_position()
        hand_pos = self._get_hand_position()
        distance = np.linalg.norm(cube_pos - hand_pos)
        
        if self.current_stage == TrainingStage.STAGE_1_APPROACH:
            # Aide pour l'approche: attirer vers le cube
            if distance > 0.1:
                direction = (cube_pos - hand_pos) / (distance + 1e-6)
                assistance = direction * assistance_level * 0.1
                # Appliquer aux actuateurs du bras
                self.data.ctrl[self.arm_actuator_ids[:3]] += assistance[:3] if len(assistance) >= 3 else assistance
        
        elif self.current_stage == TrainingStage.STAGE_2_CONTACT:
            # Aide pour le contact: guidance fine
            if distance < 0.08:
                contacts = self._get_contact_count()
                if contacts == 0:
                    # Encourager la fermeture des doigts
                    self.data.ctrl[self.hand_actuator_ids] += assistance_level * 0.2
        
        elif self.current_stage >= TrainingStage.STAGE_3_GRASP:
            # Aide pour la saisie: stabilisation
            contacts = self._get_contact_count()
            if contacts >= 2:
                # Stabiliser la saisie
                self.data.ctrl[self.hand_actuator_ids] += assistance_level * 0.3
    
    def _calculate_sophisticated_reward(self) -> float:
        """Calcul sophistiqué des récompenses selon l'étape"""
        stage_config = self.get_current_stage_config()
        weights = stage_config.reward_weights
        
        # Composants de récompense
        approach_reward = self._calculate_approach_reward() * weights.get('approach', 0)
        contact_reward = self._calculate_contact_reward() * weights.get('contact', 0)
        grasp_reward = self._calculate_grasp_reward() * weights.get('grasp', 0)
        lift_reward = self._calculate_lift_reward() * weights.get('lift', 0)
        stability_reward = self._calculate_stability_reward() * weights.get('stability', 0)
        
        # Bonus spéciaux selon l'étape
        stage_bonus = self._calculate_stage_bonus()
        
        # Pénalités
        penalties = self._calculate_penalties()
        
        total_reward = (
            approach_reward + contact_reward + grasp_reward + 
            lift_reward + stability_reward + stage_bonus - penalties
        )
        
        # Normalisation et limitation
        total_reward = np.clip(total_reward, -20.0, 100.0)
        
        return float(total_reward)
    
    def _calculate_approach_reward(self) -> float:
        """Récompense d'approche sophistiquée"""
        cube_pos = self._get_cube_position()
        hand_pos = self._get_hand_position()
        distance = np.linalg.norm(cube_pos - hand_pos)
        
        # Récompense inversement proportionnelle à la distance
        approach_reward = 10.0 / (1.0 + 5.0 * distance)
        
        # Bonus pour progression
        if distance < self.best_distance:
            self.best_distance = distance
            approach_reward += 2.0
        
        # Bonus pour proximité
        if distance < 0.05:
            approach_reward += 15.0
        elif distance < 0.1:
            approach_reward += 8.0
        elif distance < 0.15:
            approach_reward += 3.0
        
        return approach_reward
    
    def _calculate_contact_reward(self) -> float:
        """Récompense de contact sophistiquée"""
        contacts = self._get_detailed_contacts()
        contact_count = len(contacts)
        
        if contact_count == 0:
            return -2.0
        
        # Récompense progressive selon le nombre de contacts
        contact_reward = contact_count * 8.0
        
        # Bonus pour contacts multiples
        if contact_count >= 2:
            contact_reward += 10.0
        if contact_count >= 3:
            contact_reward += 15.0
        
        # Qualité des contacts
        if contacts:
            forces = [c['force'] for c in contacts]
            avg_force = np.mean(forces)
            if 0.1 < avg_force < 2.0:  # Force optimale
                contact_reward += 5.0
        
        return contact_reward
    
    def _calculate_grasp_reward(self) -> float:
        """Récompense de saisie sophistiquée"""
        contacts = self._get_detailed_contacts()
        contact_count = len(contacts)
        
        if contact_count < 2:
            return -1.0
        
        cube_pos = self._get_cube_position()
        cube_vel = self._get_cube_velocity()
        cube_speed = np.linalg.norm(cube_vel)
        
        # Récompense de base pour saisie
        grasp_reward = 15.0
        
        # Bonus pour stabilité du cube
        if cube_speed < 0.05:
            grasp_reward += 20.0
        elif cube_speed < 0.1:
            grasp_reward += 10.0
        
        # Bonus pour configuration optimale des doigts
        finger_config_bonus = self._calculate_finger_configuration_bonus()
        grasp_reward += finger_config_bonus
        
        # Bonus pour maintien de la saisie
        if len(self.contact_history) > 10:
            recent_contacts = self.contact_history[-10:]
            if all(c >= 2 for c in recent_contacts):
                grasp_reward += 15.0
        
        return grasp_reward
    
    def _calculate_lift_reward(self) -> float:
        """Récompense de levage sophistiquée"""
        if self.cube_initial_pos is None:
            return 0.0
        
        cube_pos = self._get_cube_position()
        lift_height = cube_pos[2] - self.cube_initial_pos[2]
        
        if lift_height <= 0:
            return -2.0
        
        # Récompense progressive pour la hauteur
        lift_reward = min(lift_height * 100.0, 30.0)
        
        # Bonus pour levage stable
        cube_vel = self._get_cube_velocity()
        if lift_height > 0.02 and np.linalg.norm(cube_vel) < 0.1:
            lift_reward += 25.0
        
        # Bonus pour maintien en hauteur
        if lift_height > 0.05:
            lift_reward += 20.0
        
        return lift_reward
    
    def _calculate_stability_reward(self) -> float:
        """Récompense de stabilité"""
        # Stabilité du robot
        joint_velocities = self.data.qvel[:self.model.nq]
        robot_stability = max(0, 5.0 - np.sum(np.abs(joint_velocities)))
        
        # Stabilité du cube
        cube_vel = self._get_cube_velocity()
        cube_stability = max(0, 3.0 - np.linalg.norm(cube_vel) * 10)
        
        return robot_stability + cube_stability
    
    def _calculate_stage_bonus(self) -> float:
        """Bonus spéciaux selon l'étape"""
        stage_bonus = 0.0
        
        if self.current_stage == TrainingStage.STAGE_1_APPROACH:
            # Bonus pour premier contact
            if self._get_contact_count() > 0 and len(self.contact_history) > 0 and self.contact_history[-1] == 0:
                stage_bonus += 10.0
        
        elif self.current_stage == TrainingStage.STAGE_2_CONTACT:
            # Bonus pour contact stable
            if len(self.contact_history) >= 5 and all(c > 0 for c in self.contact_history[-5:]):
                stage_bonus += 8.0
        
        elif self.current_stage == TrainingStage.STAGE_3_GRASP:
            # Bonus pour saisie réussie
            if self._is_successful_grasp():
                stage_bonus += 15.0
        
        elif self.current_stage == TrainingStage.STAGE_4_LIFT:
            # Bonus pour levage réussi
            if self._is_successful_lift():
                stage_bonus += 20.0
        
        return stage_bonus
    
    def _calculate_penalties(self) -> float:
        """Calcul des pénalités"""
        penalties = 0.0
        
        # Pénalité temporelle
        penalties += 0.01
        
        # Pénalité pour vitesses excessives
        max_joint_vel = np.max(np.abs(self.data.qvel))
        if max_joint_vel > 3.0:
            penalties += (max_joint_vel - 3.0) * 2.0
        
        # Pénalité pour actions excessives
        max_action = np.max(np.abs(self.data.ctrl))
        if max_action > 0.8:
            penalties += (max_action - 0.8) * 5.0
        
        # Pénalité pour perte du cube
        cube_pos = self._get_cube_position()
        if cube_pos[2] < 0.3:
            penalties += 10.0
        
        return penalties
    
    def _calculate_finger_configuration_bonus(self) -> float:
        """Bonus pour configuration optimale des doigts"""
        contacts = self._get_detailed_contacts()
        
        # Vérifier la diversité des contacts
        finger_types = set()
        for contact in contacts:
            if 'thumb' in contact['geom']:
                finger_types.add('thumb')
            elif 'index' in contact['geom']:
                finger_types.add('index')
            elif 'middle' in contact['geom']:
                finger_types.add('middle')
        
        # Bonus pour configuration diverse
        if 'thumb' in finger_types and len(finger_types) >= 2:
            return 10.0  # Configuration pouce + autres doigts
        elif len(finger_types) >= 2:
            return 5.0   # Configuration multi-doigts
        
        return 0.0
    
    def _get_professional_observation(self) -> np.ndarray:
        """Observation professionnelle sophistiquée"""
        # État du robot
        robot_state = np.concatenate([self.data.qpos, self.data.qvel])
        
        # Position et orientation du cube
        cube_pos = self._get_cube_position()
        cube_quat = self._get_cube_orientation()
        cube_state = np.concatenate([cube_pos, cube_quat])
        
        # Position et orientation de la main
        hand_pos = self._get_hand_position()
        hand_quat = self._get_hand_orientation()
        hand_state = np.concatenate([hand_pos, hand_quat])
        
        # Relations spatiales
        relative_pos = cube_pos - hand_pos
        distance = np.linalg.norm(relative_pos)
        spatial_info = np.array([distance, relative_pos[0], relative_pos[1], relative_pos[2]])
        
        # Informations de contact
        contacts = self._get_detailed_contacts()
        contact_count = len(contacts)
        avg_force = np.mean([c['force'] for c in contacts]) if contacts else 0.0
        max_force = np.max([c['force'] for c in contacts]) if contacts else 0.0
        
        # Qualité de la saisie
        grasp_quality = self._calculate_grasp_quality()
        is_lifted = 1.0 if self._is_cube_lifted() else 0.0
        
        contact_info = np.array([contact_count, avg_force, max_force, grasp_quality, is_lifted, 0.0])
        
        # Informations de curriculum
        stage_progress = self.stage_episodes / max(self.get_current_stage_config().episodes_for_advancement, 1)
        avg_stage_reward = np.mean(self.stage_rewards[-10:]) if self.stage_rewards else 0.0
        curriculum_info = np.array([
            self.current_stage.value[0] if hasattr(self.current_stage.value, '__getitem__') else 0.0,
            stage_progress,
            avg_stage_reward,
            len(self.stage_rewards)
        ])
        
        # Observation complète
        observation = np.concatenate([
            robot_state,      # État robot
            cube_state,       # État cube
            hand_state,       # État main
            spatial_info,     # Relations spatiales
            contact_info,     # Informations contact
            curriculum_info   # Informations curriculum
        ])
        
        # Validation finale
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation.astype(np.float32)
    
    def _get_professional_info(self) -> Dict:
        """Informations professionnelles détaillées"""
        cube_pos = self._get_cube_position()
        hand_pos = self._get_hand_position()
        contacts = self._get_detailed_contacts()
        
        info = {
            # Informations de base
            'step': self.current_step,
            'episode': self.episode_count,
            'distance': float(np.linalg.norm(cube_pos - hand_pos)),
            
            # Informations de curriculum
            'current_stage': self.current_stage.value,
            'stage_episodes': self.stage_episodes,
            'stage_progress': self.stage_episodes / max(self.get_current_stage_config().episodes_for_advancement, 1),
            
            # Performance
            'contact_count': len(contacts),
            'grasp_quality': float(self._calculate_grasp_quality()),
            'is_lifted': self._is_cube_lifted(),
            'successful_grasp': self._is_successful_grasp(),
            
            # Positions
            'cube_position': cube_pos.tolist(),
            'hand_position': hand_pos.tolist(),
            
            # Métriques avancées
            'best_distance': float(self.best_distance),
            'avg_stage_reward': float(np.mean(self.stage_rewards[-10:])) if self.stage_rewards else 0.0,
        }
        
        return info
    
    # Méthodes utilitaires professionnelles
    
    def _get_cube_position(self) -> np.ndarray:
        """Position du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id].copy()
        return np.zeros(3)
    
    def _get_cube_orientation(self) -> np.ndarray:
        """Orientation du cube (quaternion)"""
        if self.cube_body_id >= 0:
            return self.data.xquat[self.cube_body_id].copy()
        return np.array([1, 0, 0, 0])
    
    def _get_cube_velocity(self) -> np.ndarray:
        """Vitesse du cube"""
        if self.cube_body_id >= 0:
            return self.data.cvel[self.cube_body_id][:3].copy()
        return np.zeros(3)
    
    def _get_hand_position(self) -> np.ndarray:
        """Position de la main"""
        if self.hand_body_id >= 0:
            return self.data.xpos[self.hand_body_id].copy()
        return np.zeros(3)
    
    def _get_hand_orientation(self) -> np.ndarray:
        """Orientation de la main (quaternion)"""
        if self.hand_body_id >= 0:
            return self.data.xquat[self.hand_body_id].copy()
        return np.array([1, 0, 0, 0])
    
    def _get_contact_count(self) -> int:
        """Nombre de contacts simples"""
        return len(self._get_detailed_contacts())
    
    def _get_detailed_contacts(self) -> List[Dict]:
        """Contacts détaillés avec informations"""
        contacts = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Vérifier si c'est un contact main-cube
            if (geom1_name and geom2_name and
                (('cube' in geom1_name and any(finger in geom2_name for finger in ['thumb', 'index', 'middle', 'ring'])) or
                 ('cube' in geom2_name and any(finger in geom1_name for finger in ['thumb', 'index', 'middle', 'ring'])))):
                
                # Calculer la force de contact
                contact_force = np.linalg.norm(contact.force)
                
                contact_info = {
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'geom': geom1_name if 'finger' in geom1_name else geom2_name,
                    'force': contact_force,
                    'position': contact.pos.copy()
                }
                contacts.append(contact_info)
        
        return contacts
    
    def _calculate_grasp_quality(self) -> float:
        """Qualité de la saisie (0-1)"""
        contacts = self._get_detailed_contacts()
        contact_count = len(contacts)
        
        if contact_count == 0:
            return 0.0
        elif contact_count == 1:
            return 0.2
        elif contact_count == 2:
            return 0.5
        elif contact_count >= 3:
            # Vérifier la diversité et la force
            forces = [c['force'] for c in contacts]
            avg_force = np.mean(forces)
            
            # Qualité basée sur force et diversité
            force_quality = min(avg_force / 1.0, 1.0)  # Normaliser force
            diversity_bonus = self._calculate_finger_configuration_bonus() / 10.0
            
            return min(0.7 + force_quality * 0.2 + diversity_bonus, 1.0)
        
        return 0.0
    
    def _is_cube_lifted(self) -> bool:
        """Vérifier si le cube est soulevé"""
        if self.cube_initial_pos is None:
            return False
        
        cube_pos = self._get_cube_position()
        lift_height = cube_pos[2] - self.cube_initial_pos[2]
        return lift_height > 0.02
    
    def _is_successful_grasp(self) -> bool:
        """Vérifier si la saisie est réussie"""
        return (self._get_contact_count() >= 2 and 
                self._calculate_grasp_quality() > 0.5)
    
    def _is_successful_lift(self) -> bool:
        """Vérifier si le levage est réussi"""
        return (self._is_successful_grasp() and 
                self._is_cube_lifted())
    
    def _check_professional_termination(self) -> bool:
        """Vérification professionnelle des conditions de terminaison"""
        stage_config = self.get_current_stage_config()
        conditions = stage_config.termination_conditions
        
        cube_pos = self._get_cube_position()
        hand_pos = self._get_hand_position()
        distance = np.linalg.norm(cube_pos - hand_pos)
        
        # Conditions générales
        if distance > conditions.get('max_distance', 1.0):
            return True
        
        if cube_pos[2] < conditions.get('min_cube_height', 0.1):
            return True
        
        if cube_pos[2] > conditions.get('max_cube_height', 1.5):
            return True
        
        # Conditions spéciales selon l'étape
        if self.current_stage == TrainingStage.STAGE_4_LIFT or self.current_stage == TrainingStage.STAGE_5_MASTERY:
            # Terminer avec succès si levage réussi pendant assez longtemps
            if self._is_successful_lift() and self.current_step > 100:
                return True
        
        return False
    
    def _update_professional_metrics(self, reward: float):
        """Mise à jour des métriques professionnelles"""
        # Historique des contacts
        self.contact_history.append(self._get_contact_count())
        if len(self.contact_history) > 100:
            self.contact_history.pop(0)
        
        # Historique de la qualité de saisie
        self.grasp_quality_history.append(self._calculate_grasp_quality())
        if len(self.grasp_quality_history) > 100:
            self.grasp_quality_history.pop(0)
        
        # Ajouter la récompense à l'historique de l'épisode
        if not hasattr(self, 'current_episode_rewards'):
            self.current_episode_rewards = []
        
        self.current_episode_rewards.append(reward)
        
        # Vérifier la progression du curriculum en fin d'épisode
        if hasattr(self, '_episode_ending') and self._episode_ending:
            episode_reward = sum(self.current_episode_rewards)
            self.stage_rewards.append(episode_reward)
            
            # Vérifier les conditions d'avancement
            if self.auto_progression:
                self._check_curriculum_progression()
            
            self.current_episode_rewards = []
    
    def _check_curriculum_progression(self):
        """Vérifier et gérer la progression du curriculum"""
        stage_config = self.get_current_stage_config()
        
        # Vérifier si on a assez d'épisodes pour évaluer
        if len(self.stage_rewards) >= stage_config.episodes_for_advancement:
            # Calculer la performance moyenne récente
            recent_rewards = self.stage_rewards[-stage_config.episodes_for_advancement:]
            avg_reward = np.mean(recent_rewards)
            
            # Vérifier le seuil de réussite
            if avg_reward >= stage_config.success_threshold:
                self.logger.info(f"✅ Seuil atteint pour {self.current_stage.value}: {avg_reward:.2f} >= {stage_config.success_threshold}")
                self.advance_to_next_stage()
            else:
                self.logger.info(f"📊 Progression {self.current_stage.value}: {avg_reward:.2f} / {stage_config.success_threshold} (épisodes: {len(recent_rewards)})")
    
    def get_training_statistics(self) -> Dict:
        """Statistiques d'entraînement détaillées"""
        return {
            'current_stage': self.current_stage.value,
            'total_episodes': self.episode_count,
            'stage_episodes': self.stage_episodes,
            'stage_rewards': self.stage_rewards.copy(),
            'curriculum_transitions': self.metrics['curriculum_transitions'].copy(),
            'avg_contact_count': float(np.mean(self.contact_history)) if self.contact_history else 0.0,
            'avg_grasp_quality': float(np.mean(self.grasp_quality_history)) if self.grasp_quality_history else 0.0,
        }
    
    def set_stage(self, stage: TrainingStage):
        """Forcer une étape spécifique (pour les tests)"""
        old_stage = self.current_stage
        self.current_stage = stage
        
        # Reset des statistiques
        self.stage_episodes = 0
        self.stage_rewards = []
        self.stage_successes = 0
        
        self.logger.info(f"🔧 Étape forcée: {old_stage.value} → {stage.value}")
    
    def render(self):
        """Rendu professionnel"""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera="main_view")
                return self.renderer.render()
            except Exception as e:
                self.logger.warning(f"Erreur rendu: {e}")
                return None
        return None
    
    def close(self):
        """Fermeture professionnelle"""
        if hasattr(self, 'temp_model_file'):
            try:
                os.unlink(self.temp_model_file)
            except:
                pass
        
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
            except:
                pass
        
        self.logger.info("🏁 Environnement professionnel fermé")


# Test de l'environnement professionnel
if __name__ == "__main__":
    print("🧪 Test de l'environnement professionnel...")
    
    env = ProfessionalGraspEnv()
    
    print(f"✅ Environnement créé - Étape: {env.current_stage.value}")
    print(f"📊 Configuration: {env.get_current_stage_config().description}")
    
    obs, info = env.reset()
    print(f"✅ Reset OK - Obs shape: {obs.shape}")
    
    # Test de quelques steps
    total_reward = 0
    for i in range(20):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 5 == 0:
            print(f"Step {i}: reward={reward:.2f}, distance={info['distance']:.3f}, "
                  f"contacts={info['contact_count']}, stage={info['current_stage']}")
        
        if terminated or truncated:
            break
    
    print(f"✅ Test réussi! Reward total: {total_reward:.2f}")
    print(f"📈 Statistiques: {env.get_training_statistics()}")
    
    env.close()
    print("✅ Environnement fermé proprement")