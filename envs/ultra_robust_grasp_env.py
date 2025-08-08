#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT DE GRASPING ULTRA-ROBUSTE AVEC CURRICULUM LEARNING
==================================================================

Environnement de grasping professionnel et ultra-stable avec:
✅ Gestion parfaite des vitesses excessives
✅ Détection robuste des joints de doigts
✅ Physique ultra-stable avec auto-correction
✅ Curriculum learning adaptatif intelligent
✅ Système de récompenses optimisé pour le grasping
✅ Gestion d'erreurs complète
✅ Rendu Mujoco simultané pendant l'entraînement
✅ Phases de grasping professionnelles (Approche -> Contact -> Grip -> Lift)

Version finale ultra-professionnelle et fonctionnelle.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional, Union
import tempfile
import warnings
import time
import threading
import logging
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings("ignore")

class GraspPhase(Enum):
    """Phases de grasping"""
    STABILIZE = 0
    APPROACH = 1
    CONTACT = 2
    GRASP = 3
    LIFT = 4
    HOLD = 5

@dataclass
class CurriculumLevel:
    """Configuration d'un niveau de curriculum"""
    name: str
    description: str
    max_phases: int
    success_threshold: float
    episodes_required: int
    max_episode_steps: int
    cube_fixed: bool
    reward_multiplier: float
    add_noise: bool = False
    cube_variations: bool = False
    velocity_limit: float = 3.0
    precision_required: bool = False

class UltraRobustGraspEnv(gym.Env):
    """
    🎯 Environnement de Grasping Ultra-Robuste
    
    Environnement professionnel avec curriculum learning adaptatif
    et gestion complète des erreurs communes en robotique.
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 render_mode: str = "human",
                 enable_curriculum: bool = True,
                 enable_mujoco_viewer: bool = True):
        
        super().__init__()
        
        # Configuration principale
        self.render_mode = render_mode
        self.model_path_str = model_path or "/home/oussema/Documents/project/results/g1_combined.xml"
        self.enable_curriculum = enable_curriculum
        self.enable_mujoco_viewer = enable_mujoco_viewer
        
        # Logger professionnel
        self._setup_logging()
        
        # Configuration du curriculum learning ultra-adaptatif
        self._setup_curriculum()
        
        # Initialisation du modèle MuJoCo avec vérifications
        self._setup_robust_model()
        
        # Identification automatique des composants
        self._identify_robot_components()
        
        # Configuration des espaces
        self._setup_observation_action_spaces()
        
        # Variables d'état du grasping
        self._initialize_grasping_state()
        
        # Système de monitoring ultra-avancé
        self._setup_monitoring()
        
        # Viewer MuJoCo en arrière-plan
        if self.enable_mujoco_viewer:
            self._start_mujoco_viewer_thread()
        
        self.logger.info("🎯 UltraRobustGraspEnv initialisé avec succès!")
        self.logger.info(f"📚 Curriculum activé: {self.enable_curriculum}")
        self.logger.info(f"🖥️ Viewer MuJoCo: {self.enable_mujoco_viewer}")
    
    def _setup_logging(self):
        """Configuration du système de logging professionnel"""
        self.logger = logging.getLogger("UltraRobustGraspEnv")
        self.logger.setLevel(logging.INFO)
        
        # Handler pour console
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_curriculum(self):
        """Configuration du curriculum learning ultra-adaptatif"""
        self.curriculum_levels = {
            1: CurriculumLevel(
                name="ARM_STABILIZATION",
                description="Apprendre la stabilisation des bras",
                max_phases=1,  # Seulement STABILIZE
                success_threshold=20.0,
                episodes_required=5,
                max_episode_steps=300,
                cube_fixed=True,
                reward_multiplier=1.0,
                velocity_limit=2.0
            ),
            2: CurriculumLevel(
                name="CUBE_APPROACH",
                description="Apprendre l'approche du cube",
                max_phases=2,  # STABILIZE + APPROACH
                success_threshold=35.0,
                episodes_required=5,
                max_episode_steps=400,
                cube_fixed=True,
                reward_multiplier=1.3,
                velocity_limit=2.5
            ),
            3: CurriculumLevel(
                name="CONTACT_LEARNING",
                description="Apprendre le contact avec le cube",
                max_phases=3,  # STABILIZE + APPROACH + CONTACT
                success_threshold=50.0,
                episodes_required=4,
                max_episode_steps=500,
                cube_fixed=False,
                reward_multiplier=1.6,
                velocity_limit=3.0
            ),
            4: CurriculumLevel(
                name="GRASPING_MASTERY",
                description="Maîtriser la préhension complète",
                max_phases=4,  # Jusqu'à GRASP
                success_threshold=70.0,
                episodes_required=4,
                max_episode_steps=600,
                cube_fixed=False,
                reward_multiplier=2.0,
                velocity_limit=3.5,
                precision_required=True
            ),
            5: CurriculumLevel(
                name="LIFT_AND_HOLD",
                description="Soulever et maintenir le cube",
                max_phases=6,  # Toutes les phases
                success_threshold=90.0,
                episodes_required=3,
                max_episode_steps=700,
                cube_fixed=False,
                reward_multiplier=2.5,
                velocity_limit=4.0,
                add_noise=True,
                cube_variations=True,
                precision_required=True
            ),
            6: CurriculumLevel(
                name="EXPERT_MASTERY",
                description="Niveau expert avec perturbations",
                max_phases=6,
                success_threshold=120.0,
                episodes_required=3,
                max_episode_steps=800,
                cube_fixed=False,
                reward_multiplier=3.0,
                velocity_limit=5.0,
                add_noise=True,
                cube_variations=True,
                precision_required=True
            )
        }
        
        # État du curriculum
        self.current_level = 1
        self.consecutive_successes = 0
        self.level_episodes = 0
        self.level_start_time = time.time()
        self.performance_history = []
        
        # Configuration des phases selon le niveau actuel
        self._update_phase_config()
    
    def _update_phase_config(self):
        """Met à jour la configuration des phases selon le niveau de curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        
        # Durées optimales des phases selon le niveau
        phase_configs = {
            1: {'STABILIZE': 300, 'APPROACH': 0, 'CONTACT': 0, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0},
            2: {'STABILIZE': 100, 'APPROACH': 300, 'CONTACT': 0, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0},
            3: {'STABILIZE': 80, 'APPROACH': 200, 'CONTACT': 220, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0},
            4: {'STABILIZE': 60, 'APPROACH': 150, 'CONTACT': 100, 'GRASP': 200, 'LIFT': 0, 'HOLD': 0},
            5: {'STABILIZE': 50, 'APPROACH': 120, 'CONTACT': 80, 'GRASP': 150, 'LIFT': 200, 'HOLD': 100},
            6: {'STABILIZE': 40, 'APPROACH': 100, 'CONTACT': 60, 'GRASP': 120, 'LIFT': 180, 'HOLD': 100}
        }
        
        self.phase_durations = phase_configs.get(
            self.current_level, phase_configs[6]
        )
        
        self.max_episode_steps = level_config.max_episode_steps
        
        self.logger.info(f"🔄 Configuration mise à jour pour niveau {self.current_level}")
        self.logger.info(f"📊 Phases actives: {level_config.max_phases}")
        self.logger.info(f"🎯 Objectif: {level_config.success_threshold:.1f} points")
    
    def _setup_robust_model(self):
        """Configuration du modèle MuJoCo avec physique ultra-robuste"""
        try:
            # Vérification de l'existence du fichier
            if not os.path.exists(self.model_path_str):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path_str}")
            
            # Changer vers le répertoire du modèle
            original_cwd = os.getcwd()
            model_dir = os.path.dirname(self.model_path_str)
            
            if model_dir:
                os.chdir(model_dir)
            
            try:
                # Lire et optimiser le fichier XML
                with open(os.path.basename(self.model_path_str), 'r') as f:
                    xml_content = f.read()
                
                # Appliquer les corrections ultra-robustes
                xml_content = self._apply_ultra_robust_physics(xml_content)
                
                # Créer fichier temporaire optimisé
                self.temp_model_path = os.path.join(
                    model_dir if model_dir else ".", 
                    f'ultra_robust_model_{int(time.time())}.xml'
                )
                
                with open(self.temp_model_path, 'w') as f:
                    f.write(xml_content)
                
                # Charger le modèle optimisé
                self.model = mujoco.MjModel.from_xml_path(
                    os.path.basename(self.temp_model_path)
                )
                self.data = mujoco.MjData(self.model)
                
                self.logger.info("✅ Modèle MuJoCo chargé avec physique ultra-robuste")
                self.logger.info(f"  - DOFs: {self.model.nv}")
                self.logger.info(f"  - Actuateurs: {self.model.nu}")
                self.logger.info(f"  - Timestep: {self.model.opt.timestep:.6f}")
                
            finally:
                # Revenir au répertoire original
                if model_dir:
                    os.chdir(original_cwd)
        
        except Exception as e:
            if 'original_cwd' in locals() and model_dir:
                os.chdir(original_cwd)
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {e}")
    
    def _apply_ultra_robust_physics(self, xml_content: str) -> str:
        """Applique des corrections physiques ultra-robustes"""
        
        # 1. Timestep ultra-stable pour éviter les instabilités
        xml_content = xml_content.replace(
            'timestep="0.0005"',
            'timestep="0.0008"'  # Légèrement plus grand pour stabilité
        )
        
        # 2. Paramètres de solveur ultra-robustes
        xml_content = xml_content.replace(
            'iterations="500"',
            'iterations="300"'  # Équilibre performance/stabilité
        )
        
        xml_content = xml_content.replace(
            'tolerance="1e-12"',
            'tolerance="1e-10"'  # Tolérance réaliste
        )
        
        # 3. Améliorer les paramètres des actuateurs pour éviter vitesses excessives
        # Réduire kp et augmenter kv pour plus de stabilité
        xml_content = xml_content.replace(
            'kp="120" kv="25"',
            'kp="80" kv="35"'  # Plus de damping, moins de raideur
        )
        
        # 4. Optimiser les paramètres des doigts
        xml_content = xml_content.replace(
            'kp="8" kv="1.5"',
            'kp="12" kv="3"'  # Plus de contrôle pour les doigts
        )
        
        xml_content = xml_content.replace(
            'kp="6" kv="1"',
            'kp="10" kv="2.5"'  # Meilleur contrôle fin
        )
        
        xml_content = xml_content.replace(
            'kp="10" kv="2"',
            'kp="15" kv="4"'  # Pouce plus réactif
        )
        
        # 5. Améliorer la friction pour un meilleur grasping
        xml_content = xml_content.replace(
            'friction="5.0 1.0 0.5"',
            'friction="3.0 0.8 0.4"'  # Friction plus réaliste
        )
        
        # 6. Ajouter des limites de vitesse dans les joints si pas présent
        # Cette partie nécessiterait une analyse plus poussée du XML
        # pour l'instant on garde les paramètres actuels qui semblent corrects
        
        return xml_content
    
    def _identify_robot_components(self):
        """Identification automatique et robuste des composants du robot"""
        try:
            # Identification des joints des bras
            self.arm_joint_ids = []
            self.arm_joint_names = []
            
            # Identification des joints des doigts
            self.finger_joint_ids = []
            self.finger_joint_names = []
            
            # Mots-clés pour identification automatique
            arm_keywords = ['shoulder', 'elbow', 'wrist']
            finger_keywords = ['finger', 'thumb', 'index', 'middle', 'ring']
            
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    joint_name_lower = joint_name.lower()
                    
                    # Vérifier si c'est un joint de bras
                    if any(keyword in joint_name_lower for keyword in arm_keywords):
                        self.arm_joint_ids.append(i)
                        self.arm_joint_names.append(joint_name)
                    
                    # Vérifier si c'est un joint de doigt
                    elif any(keyword in joint_name_lower for keyword in finger_keywords):
                        self.finger_joint_ids.append(i)
                        self.finger_joint_names.append(joint_name)
            
            # Identification du cube
            try:
                self.cube_body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
                )
                if self.cube_body_id < 0:
                    raise ValueError("Cube body not found")
            except:
                self.logger.warning("⚠️ Corps 'cube' non trouvé, recherche alternative...")
                # Recherche alternative pour le cube
                self.cube_body_id = -1
                for i in range(self.model.nbody):
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and 'cube' in body_name.lower():
                        self.cube_body_id = i
                        break
            
            # Identification des sites des doigts pour détection de contact
            self.finger_sites = []
            self.finger_site_names = []
            
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name:
                    site_name_lower = site_name.lower()
                    if any(keyword in site_name_lower for keyword in ['tip', 'finger', 'thumb']):
                        self.finger_sites.append(i)
                        self.finger_site_names.append(site_name)
            
            # Vérifications et logging
            self.logger.info("✅ Composants robot identifiés:")
            self.logger.info(f"  - Joints bras: {len(self.arm_joint_ids)} {self.arm_joint_names}")
            self.logger.info(f"  - Joints doigts: {len(self.finger_joint_ids)} {self.finger_joint_names}")
            self.logger.info(f"  - Sites doigts: {len(self.finger_sites)} {self.finger_site_names}")
            self.logger.info(f"  - Cube body ID: {self.cube_body_id}")
            
            # Vérifications critiques
            if len(self.arm_joint_ids) == 0:
                raise ValueError("Aucun joint de bras trouvé!")
            
            if len(self.finger_joint_ids) == 0:
                self.logger.warning("⚠️ Aucun joint de doigt trouvé!")
            
            if self.cube_body_id < 0:
                self.logger.warning("⚠️ Cube non trouvé, utilisation position par défaut")
        
        except Exception as e:
            self.logger.error(f"❌ Erreur identification composants: {e}")
            raise RuntimeError(f"Échec identification composants: {e}")
    
    def _setup_observation_action_spaces(self):
        """Configuration des espaces d'action et d'observation"""
        # Espace d'action: tous les actuateurs du modèle
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Calculer la taille d'observation de manière robuste
        obs_size = self._calculate_observation_size()
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        self.logger.info(f"🎮 Espaces configurés:")
        self.logger.info(f"  - Actions: {self.action_space.shape}")
        self.logger.info(f"  - Observations: {self.observation_space.shape}")
    
    def _calculate_observation_size(self) -> int:
        """Calcule la taille exacte de l'observation"""
        size = 0
        
        # Positions et vitesses des joints
        size += self.model.nq  # Positions
        size += self.model.nv  # Vitesses
        
        # Position et orientation du cube (7)
        size += 7
        
        # Informations de phase et curriculum (8)
        size += 8
        
        # Informations de grasping (10)
        size += 10
        
        return size
    
    def _initialize_grasping_state(self):
        """Initialise les variables d'état du grasping"""
        # Phase actuelle
        self.current_phase = GraspPhase.STABILIZE
        self.phase_timer = 0
        self.episode_step = 0
        
        # État du cube et positions
        self.cube_initial_pos = np.array([0.3, 0.0, 0.05])
        self.cube_target_lifted_height = 0.12
        
        # Compteurs et états de grasping
        self.stability_count = 0
        self.contact_count = 0
        self.grasp_strength = 0.0
        self.successful_grasp = False
        self.cube_lifted = False
        self.hold_duration = 0
        
        # Historiques pour analyse de stabilité
        self.velocity_history = []
        self.position_history = []
        self.contact_history = []
        self.max_history_length = 30
        
        # Métriques de performance
        self.phase_success_count = {phase: 0 for phase in GraspPhase}
        self.total_contact_time = 0
        self.best_lift_height = 0.0
    
    def _setup_monitoring(self):
        """Configuration du système de monitoring avancé"""
        self.monitoring = {
            'velocities': [],
            'rewards': [],
            'phases': [],
            'contacts': [],
            'cube_positions': [],
            'episode_stats': [],
            'curriculum_transitions': [],
            'max_velocity_violations': 0,
            'stability_violations': 0,
            'contact_failures': 0
        }
    
    def _start_mujoco_viewer_thread(self):
        """Démarre le viewer MuJoCo en thread séparé"""
        def run_viewer():
            try:
                import mujoco.viewer
                self.logger.info("🖥️ Démarrage viewer MuJoCo...")
                
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    self.mujoco_viewer = viewer
                    viewer.cam.distance = 1.5
                    viewer.cam.azimuth = 45
                    viewer.cam.elevation = -20
                    viewer.cam.lookat = [0.3, 0.0, 0.1]
                    
                    while True:
                        try:
                            viewer.sync()
                            time.sleep(0.01)
                        except:
                            break
                            
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur viewer MuJoCo: {e}")
        
        self.viewer_thread = threading.Thread(target=run_viewer, daemon=True)
        self.viewer_thread.start()
        time.sleep(1)  # Laisser le temps au viewer de s'initialiser
    
    def reset(self, seed=None, options=None):
        """Reset ultra-robuste de l'environnement"""
        if seed is not None:
            np.random.seed(seed)
        
        try:
            # Reset du modèle MuJoCo
            mujoco.mj_resetData(self.model, self.data)
            
            # Configuration du cube selon le curriculum
            self._configure_cube_position()
            
            # Configuration initiale des bras
            self._configure_initial_arm_positions()
            
            # Reset des variables d'état
            self._reset_grasping_state()
            
            # Stabilisation initiale
            self._initial_stabilization()
            
            # Première observation
            observation = self._get_observation()
            info = self._get_info()
            
            self.logger.debug(f"🔄 Reset terminé - Niveau {self.current_level} - Phase {self.current_phase.name}")
            
            return observation, info
            
        except Exception as e:
            self.logger.error(f"❌ Erreur reset: {e}")
            # Reset d'urgence
            mujoco.mj_resetData(self.model, self.data)
            return self._get_observation(), self._get_info()
    
    def _configure_cube_position(self):
        """Configure la position du cube selon le curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        
        if level_config.cube_variations:
            # Position variable pour niveaux avancés
            offset = np.random.uniform(-0.08, 0.08, 3)
            offset[2] = abs(offset[2]) * 0.5  # Garder Z positif et petit
        else:
            # Position fixe ou légèrement variable
            offset = np.random.uniform(-0.02, 0.02, 3)
            offset[2] = abs(offset[2]) * 0.3
        
        self.cube_initial_pos = np.array([0.3, 0.0, 0.05]) + offset
        
        if self.cube_body_id >= 0:
            # Définir position et orientation du cube
            cube_qpos_start = self.model.nq - 7  # Les 7 derniers qpos pour le cube (pos + quat)
            self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = self.cube_initial_pos
            # Orientation identité (pas de rotation)
            self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
    
    def _configure_initial_arm_positions(self):
        """Configure les positions initiales des bras"""
        level_config = self.curriculum_levels[self.current_level]
        
        # Position de départ optimisée pour le grasping
        if len(self.arm_joint_ids) >= 14:  # 2 bras × 7 joints
            # Positions neutres optimisées
            left_arm_pos = [0.0, 0.2, 0.0, -0.8, 0.0, 0.6, 0.0]  # Bras gauche
            right_arm_pos = [0.0, -0.2, 0.0, -0.8, 0.0, 0.6, 0.0]  # Bras droit
            
            arm_positions = left_arm_pos + right_arm_pos
            
            # Ajouter du bruit pour niveaux avancés
            if level_config.add_noise:
                noise = np.random.uniform(-0.1, 0.1, len(arm_positions))
                arm_positions = [pos + n for pos, n in zip(arm_positions, noise)]
            
            # Appliquer les positions
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(arm_positions):
                    self.data.qpos[joint_id] = arm_positions[i]
        
        # Configuration des doigts (ouverts au début)
        for joint_id in self.finger_joint_ids:
            self.data.qpos[joint_id] = 0.1  # Légèrement ouverts
    
    def _reset_grasping_state(self):
        """Reset des variables d'état du grasping"""
        self.current_phase = GraspPhase.STABILIZE
        self.phase_timer = 0
        self.episode_step = 0
        
        self.stability_count = 0
        self.contact_count = 0
        self.grasp_strength = 0.0
        self.successful_grasp = False
        self.cube_lifted = False
        self.hold_duration = 0
        
        # Reset des historiques
        self.velocity_history.clear()
        self.position_history.clear()
        self.contact_history.clear()
        
        # Reset métriques
        self.total_contact_time = 0
        self.best_lift_height = 0.0
    
    def _initial_stabilization(self):
        """Stabilisation initiale du système"""
        for _ in range(20):
            # Actions nulles pour stabiliser
            zero_actions = np.zeros(self.model.nu)
            self.data.ctrl[:] = zero_actions
            mujoco.mj_step(self.model, self.data)
    
    def step(self, action):
        """Step ultra-robuste avec gestion complète des erreurs"""
        try:
            self.episode_step += 1
            self.phase_timer += 1
            
            # Validation et limitation des actions
            action = self._validate_and_limit_actions(action)
            
            # Application des actions avec contrôle de vitesse
            self._apply_controlled_actions(action)
            
            # Simulation physique avec vérifications
            self._safe_physics_step()
            
            # Gestion des phases de grasping
            self._update_grasping_phase()
            
            # Calcul de l'observation et récompense
            observation = self._get_observation()
            reward = self._calculate_advanced_reward()
            
            # Vérification des conditions de terminaison
            terminated = self._check_termination()
            truncated = self.episode_step >= self.max_episode_steps
            
            # Informations de debug
            info = self._get_info()
            
            # Monitoring
            self._update_monitoring(action, reward, observation)
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            self.logger.error(f"❌ Erreur critique dans step(): {e}")
            # Retour d'urgence
            return (self._get_observation(), 
                   -10.0, 
                   True, 
                   True, 
                   {'error': str(e)})
    
    def _validate_and_limit_actions(self, action) -> np.ndarray:
        """Valide et limite les actions pour éviter les problèmes"""
        # Convertir en numpy array
        action = np.asarray(action, dtype=np.float32)
        
        # Vérifier la taille
        if len(action) != self.model.nu:
            self.logger.warning(f"⚠️ Taille action incorrecte: {len(action)} vs {self.model.nu}")
            # Ajuster la taille
            if len(action) < self.model.nu:
                action = np.pad(action, (0, self.model.nu - len(action)))
            else:
                action = action[:self.model.nu]
        
        # Limiter les valeurs
        action = np.clip(action, -1.0, 1.0)
        
        # Vérifier les NaN/Inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("⚠️ Actions NaN/Inf détectées, reset à zéro")
            action = np.zeros(self.model.nu, dtype=np.float32)
        
        return action
    
    def _apply_controlled_actions(self, action):
        """Applique les actions avec contrôle de vitesse avancé"""
        level_config = self.curriculum_levels[self.current_level]
        
        # Scaling adaptatif selon la phase et le niveau
        phase_scalings = {
            GraspPhase.STABILIZE: 0.02,
            GraspPhase.APPROACH: 0.08,
            GraspPhase.CONTACT: 0.05,
            GraspPhase.GRASP: 0.03,
            GraspPhase.LIFT: 0.06,
            GraspPhase.HOLD: 0.01
        }
        
        base_scaling = phase_scalings.get(self.current_phase, 0.05)
        
        # Ajustement selon le niveau de curriculum
        level_multipliers = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.85, 5: 1.0, 6: 1.2}
        curriculum_multiplier = level_multipliers.get(self.current_level, 1.0)
        
        final_scaling = base_scaling * curriculum_multiplier
        
        # Ajouter du bruit pour niveaux avancés
        if level_config.add_noise:
            noise = np.random.normal(0, 0.005, action.shape)
            action = action + noise
        
        # Scaling différencié bras/doigts
        num_arm_joints = len(self.arm_joint_ids)
        num_finger_joints = len(self.finger_joint_ids)
        
        # Actions pour les bras
        if num_arm_joints > 0:
            arm_actions = action[:num_arm_joints] * final_scaling
            
            # Application avec contrôle de vitesse
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(arm_actions):
                    current_pos = self.data.qpos[joint_id]
                    target_pos = current_pos + arm_actions[i]
                    
                    # Limiter le changement maximal
                    max_change = 0.02 if self.current_level <= 2 else 0.04
                    if abs(target_pos - current_pos) > max_change:
                        target_pos = current_pos + np.sign(target_pos - current_pos) * max_change
                    
                    # Appliquer les limites du joint
                    joint_range = self.model.jnt_range[joint_id]
                    if joint_range[0] < joint_range[1]:  # Range valide
                        target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                    
                    self.data.ctrl[i] = target_pos
        
        # Actions pour les doigts
        if num_finger_joints > 0:
            finger_start_idx = num_arm_joints
            finger_actions = action[finger_start_idx:finger_start_idx + num_finger_joints]
            
            # Scaling spécial pour les doigts selon la phase
            finger_scaling = final_scaling * 2.0
            if self.current_phase == GraspPhase.GRASP:
                finger_scaling *= 3.0  # Plus réactif en phase de grasp
            
            finger_actions = finger_actions * finger_scaling
            
            for i, joint_id in enumerate(self.finger_joint_ids):
                if i < len(finger_actions):
                    current_pos = self.data.qpos[joint_id]
                    target_pos = current_pos + finger_actions[i]
                    
                    # Limites réalistes pour les doigts
                    target_pos = np.clip(target_pos, 0.0, 1.4)
                    
                    ctrl_idx = num_arm_joints + i
                    if ctrl_idx < self.model.nu:
                        self.data.ctrl[ctrl_idx] = target_pos
    
    def _safe_physics_step(self):
        """Simulation physique avec vérifications de sécurité"""
        try:
            # Sauvegarder l'état avant simulation
            qpos_backup = self.data.qpos.copy()
            qvel_backup = self.data.qvel.copy()
            
            # Simulation
            mujoco.mj_step(self.model, self.data)
            
            # Vérifications post-simulation
            self._check_and_correct_physics_violations(qpos_backup, qvel_backup)
            
        except Exception as e:
            self.logger.error(f"❌ Erreur simulation physique: {e}")
            # Restaurer état précédent
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
    
    def _check_and_correct_physics_violations(self, qpos_backup, qvel_backup):
        """Vérifie et corrige les violations physiques"""
        level_config = self.curriculum_levels[self.current_level]
        
        # 1. Vérifier NaN/Inf
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
            np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            
            self.logger.warning("⚠️ NaN/Inf détecté, restauration état précédent")
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            self.monitoring['stability_violations'] += 1
            return
        
        # 2. Contrôle des vitesses excessives
        max_velocity = np.max(np.abs(self.data.qvel))
        velocity_limit = level_config.velocity_limit
        
        if max_velocity > velocity_limit:
            # Réduction progressive des vitesses
            reduction_factor = velocity_limit / max_velocity
            self.data.qvel *= reduction_factor
            
            self.monitoring['max_velocity_violations'] += 1
            
            if self.episode_step % 100 == 0:
                self.logger.debug(f"⚡ Vitesse excessive corrigée: {max_velocity:.2f} -> {velocity_limit:.2f}")
        
        # 3. Mise à jour historique de stabilité
        arm_velocities = []
        for joint_id in self.arm_joint_ids:
            if joint_id < len(self.data.qvel):
                arm_velocities.append(abs(self.data.qvel[joint_id]))
        
        if arm_velocities:
            mean_arm_velocity = np.mean(arm_velocities)
            self.velocity_history.append(mean_arm_velocity)
            
            if len(self.velocity_history) > self.max_history_length:
                self.velocity_history.pop(0)
            
            # Compter stabilité
            stability_threshold = 0.15 if self.current_level <= 3 else 0.2
            if mean_arm_velocity < stability_threshold:
                self.stability_count += 1
            else:
                self.stability_count = max(0, self.stability_count - 2)
    
    def _update_grasping_phase(self):
        """Gestion intelligente des phases de grasping"""
        level_config = self.curriculum_levels[self.current_level]
        max_phases = level_config.max_phases
        
        # Ne pas dépasser le niveau autorisé
        if self.current_phase.value >= max_phases:
            return
        
        should_advance = self._check_phase_advancement()
        
        if should_advance and self.current_phase.value < min(5, max_phases - 1):
            self.current_phase = GraspPhase(self.current_phase.value + 1)
            self.phase_timer = 0
            
            self.phase_success_count[self.current_phase] += 1
            
            if self.episode_step % 50 == 0:
                self.logger.debug(f"📈 Transition vers: {self.current_phase.name}")
    
    def _check_phase_advancement(self) -> bool:
        """Vérifie si on peut avancer à la phase suivante"""
        phase_duration = self.phase_durations.get(self.current_phase.name, 100)
        
        if self.current_phase == GraspPhase.STABILIZE:
            # Critères de stabilité adaptatifs
            stability_threshold = 20 if self.current_level <= 2 else 30
            return (self.stability_count > stability_threshold or 
                   self.phase_timer >= phase_duration)
        
        elif self.current_phase == GraspPhase.APPROACH:
            # Critères d'approche
            cube_pos = self._get_cube_position()
            hand_center = self._get_hand_center()
            distance = np.linalg.norm(cube_pos - hand_center)
            
            distance_threshold = 0.25 if self.current_level <= 2 else 0.15
            return (distance < distance_threshold or 
                   self.phase_timer >= phase_duration)
        
        elif self.current_phase == GraspPhase.CONTACT:
            # Critères de contact
            contact_detected = self._detect_robust_contact()
            return (contact_detected or self.phase_timer >= phase_duration)
        
        elif self.current_phase == GraspPhase.GRASP:
            # Critères de préhension
            grasp_stable = self._check_grasp_stability()
            return (grasp_stable or self.phase_timer >= phase_duration)
        
        elif self.current_phase == GraspPhase.LIFT:
            # Critères de soulèvement
            cube_lifted = self._is_cube_lifted()
            return (cube_lifted or self.phase_timer >= phase_duration)
        
        elif self.current_phase == GraspPhase.HOLD:
            # Phase de maintien
            return self.phase_timer >= phase_duration
        
        return False
    
    def _get_cube_position(self) -> np.ndarray:
        """Obtient la position du cube de manière robuste"""
        if self.cube_body_id >= 0 and self.cube_body_id < self.model.nbody:
            return self.data.xpos[self.cube_body_id].copy()
        else:
            # Position par défaut si cube non trouvé
            return self.cube_initial_pos.copy()
    
    def _get_hand_center(self) -> np.ndarray:
        """Calcule le centre des mains de manière robuste"""
        if len(self.finger_sites) > 0:
            valid_positions = []
            for site_id in self.finger_sites:
                if site_id < self.model.nsite:
                    pos = self.data.site_xpos[site_id]
                    if not np.any(np.isnan(pos)):
                        valid_positions.append(pos)
            
            if valid_positions:
                return np.mean(valid_positions, axis=0)
        
        # Fallback: estimation basée sur les positions des bras
        # Utiliser les positions des poignets comme approximation
        wrist_positions = []
        for i, joint_id in enumerate(self.arm_joint_ids):
            joint_name = self.arm_joint_names[i] if i < len(self.arm_joint_names) else ""
            if 'wrist' in joint_name.lower():
                # Estimation grossière de la position du poignet
                # (nécessiterait une forward kinematics complète pour être précise)
                pass
        
        # Position par défaut devant le robot
        return np.array([0.4, 0.0, 0.1])
    
    def _detect_robust_contact(self) -> bool:
        """Détection de contact robuste et multi-méthodes"""
        contact_methods = []
        
        # Méthode 1: Distance géométrique
        cube_pos = self._get_cube_position()
        hand_center = self._get_hand_center()
        geometric_distance = np.linalg.norm(cube_pos - hand_center)
        
        contact_threshold = 0.08
        geometric_contact = geometric_distance < contact_threshold
        contact_methods.append(geometric_contact)
        
        # Méthode 2: Contacts physiques MuJoCo
        physical_contacts = 0
        if self.cube_body_id >= 0:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.dist < 0.01:  # Contact réel
                    body1 = self.model.geom_bodyid[contact.geom1]
                    body2 = self.model.geom_bodyid[contact.geom2]
                    
                    if body1 == self.cube_body_id or body2 == self.cube_body_id:
                        # Vérifier si l'autre corps est une main/doigt
                        other_body = body2 if body1 == self.cube_body_id else body1
                        if other_body < self.model.nbody:
                            body_name = mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_BODY, other_body
                            ) or ""
                            
                            if any(kw in body_name.lower() for kw in 
                                  ['finger', 'thumb', 'palm', 'hand']):
                                physical_contacts += 1
        
        physical_contact = physical_contacts > 0
        contact_methods.append(physical_contact)
        
        # Méthode 3: Sites de doigts proches
        finger_contacts = 0
        for site_id in self.finger_sites:
            if site_id < self.model.nsite:
                finger_pos = self.data.site_xpos[site_id]
                distance = np.linalg.norm(cube_pos - finger_pos)
                if distance < 0.06:
                    finger_contacts += 1
        
        sites_contact = finger_contacts >= 2
        contact_methods.append(sites_contact)
        
        # Contact détecté si au moins 2 méthodes sont positives
        contact_detected = sum(contact_methods) >= 2
        
        # Mise à jour historique
        self.contact_history.append(contact_detected)
        if len(self.contact_history) > self.max_history_length:
            self.contact_history.pop(0)
        
        # Compter le temps de contact
        if contact_detected:
            self.contact_count += 1
            self.total_contact_time += 1
        
        return contact_detected
    
    def _check_grasp_stability(self) -> bool:
        """Vérifie la stabilité de la préhension"""
        if not self._detect_robust_contact():
            return False
        
        # Vérifier la vitesse du cube
        if self.cube_body_id >= 0:
            cube_vel = self.data.cvel[self.cube_body_id]
            cube_speed = np.linalg.norm(cube_vel)
            
            velocity_stable = cube_speed < 0.1
        else:
            velocity_stable = True
        
        # Vérifier la fermeture des doigts
        finger_closure = self._calculate_finger_closure()
        closure_adequate = finger_closure > 0.3
        
        # Vérifier la durée de contact
        contact_duration = sum(self.contact_history[-10:]) if len(self.contact_history) >= 10 else 0
        duration_adequate = contact_duration >= 7
        
        grasp_stable = velocity_stable and closure_adequate and duration_adequate
        
        if grasp_stable:
            self.successful_grasp = True
            self.grasp_strength = min(1.0, self.grasp_strength + 0.1)
        else:
            self.grasp_strength = max(0.0, self.grasp_strength - 0.05)
        
        return grasp_stable
    
    def _calculate_finger_closure(self) -> float:
        """Calcule le degré de fermeture des doigts"""
        if len(self.finger_joint_ids) == 0:
            return 0.5  # Valeur par défaut
        
        closure_values = []
        for joint_id in self.finger_joint_ids:
            if joint_id < len(self.data.qpos):
                joint_pos = self.data.qpos[joint_id]
                # Normaliser entre 0 (ouvert) et 1 (fermé)
                normalized_closure = max(0.0, min(1.0, joint_pos / 1.2))
                closure_values.append(normalized_closure)
        
        return np.mean(closure_values) if closure_values else 0.0
    
    def _is_cube_lifted(self) -> bool:
        """Vérifie si le cube est soulevé"""
        cube_pos = self._get_cube_position()
        lift_height = cube_pos[2] - self.cube_initial_pos[2]
        
        lifted = lift_height > 0.04
        
        if lifted:
            self.cube_lifted = True
            self.best_lift_height = max(self.best_lift_height, lift_height)
        
        return lifted
    
    def _calculate_advanced_reward(self) -> float:
        """Système de récompenses avancé et adaptatif"""
        level_config = self.curriculum_levels[self.current_level]
        reward = 0.0
        
        # Récompense de survie (encourage à continuer)
        reward += 0.5 * level_config.reward_multiplier
        
        # Pénalités pour vitesses excessives
        max_velocity = np.max(np.abs(self.data.qvel))
        if max_velocity > level_config.velocity_limit:
            reward -= (max_velocity - level_config.velocity_limit) * 2.0
        
        # Bonus de stabilité progressif
        if len(self.velocity_history) > 0:
            recent_stability = np.mean(self.velocity_history[-5:])
            if recent_stability < 0.1:
                reward += 2.0 * level_config.reward_multiplier
            elif recent_stability < 0.2:
                reward += 1.0 * level_config.reward_multiplier
        
        # Récompenses spécifiques selon la phase
        reward += self._calculate_phase_rewards(level_config)
        
        # Récompenses de progression de phase
        reward += self.current_phase.value * 3.0 * level_config.reward_multiplier
        
        # Bonus de curriculum avancé
        if level_config.precision_required:
            precision_bonus = self._calculate_precision_bonus()
            reward += precision_bonus * level_config.reward_multiplier
        
        # Pénalités pour échecs critiques
        reward += self._calculate_failure_penalties()
        
        # Bonus de performance globale
        reward += self._calculate_performance_bonus(level_config)
        
        # Limiter la récompense dans un range raisonnable
        reward = np.clip(reward, -50.0, 200.0)
        
        return float(reward)
    
    def _calculate_phase_rewards(self, level_config) -> float:
        """Calcule les récompenses spécifiques à chaque phase"""
        reward = 0.0
        multiplier = level_config.reward_multiplier
        
        if self.current_phase == GraspPhase.STABILIZE:
            # Récompenser la stabilité des bras
            if self.stability_count > 0:
                reward += min(self.stability_count * 0.2, 5.0) * multiplier
            
            # Bonus pour maintien prolongé de la stabilité
            if self.stability_count > 40:
                reward += 8.0 * multiplier
        
        elif self.current_phase == GraspPhase.APPROACH:
            # Récompenser l'approche du cube
            cube_pos = self._get_cube_position()
            hand_center = self._get_hand_center()
            distance = np.linalg.norm(cube_pos - hand_center)
            
            # Récompense inversement proportionnelle à la distance
            approach_reward = max(0, (0.5 - distance) / 0.5) * 10.0
            reward += approach_reward * multiplier
            
            # Bonus pour approche directe et smooth
            if distance < 0.2:
                reward += 15.0 * multiplier
            if distance < 0.1:
                reward += 25.0 * multiplier
        
        elif self.current_phase == GraspPhase.CONTACT:
            # Récompenser le contact avec le cube
            if self._detect_robust_contact():
                reward += 12.0 * multiplier
                
                # Bonus pour contact prolongé
                recent_contacts = sum(self.contact_history[-5:]) if len(self.contact_history) >= 5 else 0
                if recent_contacts >= 4:
                    reward += 8.0 * multiplier
        
        elif self.current_phase == GraspPhase.GRASP:
            # Récompenser la préhension stable
            if self._check_grasp_stability():
                reward += 20.0 * multiplier
            
            # Récompenser la fermeture des doigts
            finger_closure = self._calculate_finger_closure()
            reward += finger_closure * 15.0 * multiplier
            
            # Bonus pour préhension maintenue
            if self.successful_grasp:
                reward += 25.0 * multiplier
        
        elif self.current_phase == GraspPhase.LIFT:
            # Récompenser le soulèvement
            cube_pos = self._get_cube_position()
            lift_height = cube_pos[2] - self.cube_initial_pos[2]
            
            if lift_height > 0:
                lift_reward = min(lift_height / 0.1, 1.0) * 30.0
                reward += lift_reward * multiplier
            
            if self._is_cube_lifted():
                reward += 40.0 * multiplier
        
        elif self.current_phase == GraspPhase.HOLD:
            # Récompenser le maintien du cube
            if (self._is_cube_lifted() and 
                self._check_grasp_stability() and 
                self._detect_robust_contact()):
                
                reward += 35.0 * multiplier
                self.hold_duration += 1
                
                # Bonus croissant pour maintien prolongé
                hold_bonus = min(self.hold_duration * 0.5, 20.0)
                reward += hold_bonus * multiplier
        
        return reward
    
    def _calculate_precision_bonus(self) -> float:
        """Calcule les bonus de précision pour niveaux avancés"""
        bonus = 0.0
        
        # Précision de l'approche
        cube_pos = self._get_cube_position()
        hand_center = self._get_hand_center()
        approach_precision = max(0, 1.0 - np.linalg.norm(cube_pos - hand_center) / 0.3)
        bonus += approach_precision * 5.0
        
        # Précision de la préhension
        if self.successful_grasp:
            grasp_precision = self.grasp_strength
            bonus += grasp_precision * 8.0
        
        # Précision du soulèvement
        if self.cube_lifted:
            cube_pos = self._get_cube_position()
            lift_stability = 1.0 - min(np.linalg.norm(self.data.cvel[self.cube_body_id]) / 2.0, 1.0)
            bonus += lift_stability * 6.0
        
        return bonus
    
    def _calculate_failure_penalties(self) -> float:
        """Calcule les pénalités pour échecs"""
        penalty = 0.0
        
        # Pénalité pour cube tombé
        cube_pos = self._get_cube_position()
        if cube_pos[2] < 0.0:
            penalty -= 30.0
        
        # Pénalité pour cube trop loin
        if (abs(cube_pos[0]) > 1.0 or abs(cube_pos[1]) > 1.0):
            penalty -= 20.0
        
        # Pénalité pour instabilité excessive
        if self.monitoring['max_velocity_violations'] > 10:
            penalty -= 5.0
        
        # Pénalité pour absence de progrès
        if (self.current_phase == GraspPhase.STABILIZE and 
            self.phase_timer > 200 and 
            self.stability_count < 10):
            penalty -= 10.0
        
        return penalty
    
    def _calculate_performance_bonus(self, level_config) -> float:
        """Calcule les bonus de performance globale"""
        bonus = 0.0
        
        # Bonus d'efficacité temporelle
        if self.current_phase.value > 0:
            phase_efficiency = max(0, 1.0 - self.phase_timer / 300.0)
            bonus += phase_efficiency * 5.0 * level_config.reward_multiplier
        
        # Bonus de progression fluide
        if self.current_phase.value >= 2:
            smoothness_bonus = min(self.current_phase.value * 2.0, 10.0)
            bonus += smoothness_bonus * level_config.reward_multiplier
        
        # Bonus de maîtrise complète (toutes les phases réussies)
        if (self.current_phase == GraspPhase.HOLD and 
            self.successful_grasp and 
            self.cube_lifted):
            bonus += 50.0 * level_config.reward_multiplier
        
        return bonus
    
    def _get_observation(self) -> np.ndarray:
        """Construit l'observation complète de l'état"""
        obs = []
        
        try:
            # 1. Positions des joints (nq)
            obs.extend(self.data.qpos.copy())
            
            # 2. Vitesses des joints (nv)
            obs.extend(self.data.qvel.copy())
            
            # 3. Position et orientation du cube (7)
            cube_pos = self._get_cube_position()
            obs.extend(cube_pos)
            
            if self.cube_body_id >= 0:
                cube_quat = self.data.xquat[self.cube_body_id].copy()
                obs.extend(cube_quat)
            else:
                obs.extend([1.0, 0.0, 0.0, 0.0])  # Quaternion identité
            
            # 4. Informations de phase et curriculum (8)
            obs.extend([
                float(self.current_phase.value),
                float(self.phase_timer / 100.0),  # Normalisé
                float(self.current_level),
                float(self.episode_step / 1000.0),  # Normalisé
                float(self.stability_count / 50.0),  # Normalisé
                float(self.contact_count / 100.0),  # Normalisé
                float(self.successful_grasp),
                float(self.cube_lifted)
            ])
            
            # 5. Informations de grasping avancées (10)
            hand_center = self._get_hand_center()
            cube_distance = np.linalg.norm(cube_pos - hand_center)
            finger_closure = self._calculate_finger_closure()
            
            obs.extend([
                cube_distance,
                finger_closure,
                self.grasp_strength,
                float(self.hold_duration / 50.0),  # Normalisé
                float(self.best_lift_height / 0.2),  # Normalisé
                float(np.mean(self.velocity_history[-5:]) if self.velocity_history else 0.0),
                float(np.mean(self.contact_history[-5:]) if self.contact_history else 0.0),
                float(self.total_contact_time / 200.0),  # Normalisé
                float(len(self.performance_history)),
                float(self.consecutive_successes / 10.0)  # Normalisé
            ])
            
            # Conversion en array et vérifications
            obs_array = np.array(obs, dtype=np.float32)
            
            # Vérifier les NaN/Inf
            if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
                self.logger.warning("⚠️ NaN/Inf dans observation, correction...")
                obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Vérifier la taille
            expected_size = self.observation_space.shape[0]
            if len(obs_array) != expected_size:
                self.logger.warning(f"⚠️ Taille observation: {len(obs_array)} vs {expected_size}")
                if len(obs_array) < expected_size:
                    padding = np.zeros(expected_size - len(obs_array), dtype=np.float32)
                    obs_array = np.concatenate([obs_array, padding])
                else:
                    obs_array = obs_array[:expected_size]
            
            return obs_array
            
        except Exception as e:
            self.logger.error(f"❌ Erreur critique dans _get_observation: {e}")
            # Observation d'urgence
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Informations détaillées de l'environnement"""
        try:
            cube_pos = self._get_cube_position()
            hand_center = self._get_hand_center()
            
            return {
                'episode_step': self.episode_step,
                'current_phase': self.current_phase.name,
                'curriculum_level': self.current_level,
                'curriculum_name': self.curriculum_levels[self.current_level].name,
                'phase_timer': self.phase_timer,
                'stability_count': self.stability_count,
                'successful_grasp': self.successful_grasp,
                'cube_lifted': self.cube_lifted,
                'cube_position': cube_pos.tolist(),
                'hand_center': hand_center.tolist(),
                'cube_distance': float(np.linalg.norm(cube_pos - hand_center)),
                'finger_closure': float(self._calculate_finger_closure()),
                'grasp_strength': float(self.grasp_strength),
                'contact_count': self.contact_count,
                'hold_duration': self.hold_duration,
                'best_lift_height': float(self.best_lift_height),
                'consecutive_successes': self.consecutive_successes,
                'max_velocity': float(np.max(np.abs(self.data.qvel))),
                'avg_arm_velocity': float(np.mean([abs(self.data.qvel[i]) for i in self.arm_joint_ids 
                                                 if i < len(self.data.qvel)])) if self.arm_joint_ids else 0.0,
                'performance_score': float(self._calculate_performance_score())
            }
        except Exception as e:
            self.logger.error(f"❌ Erreur dans _get_info: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_score(self) -> float:
        """Calcule un score de performance global"""
        try:
            score = 0.0
            
            # Score de base selon la phase atteinte
            phase_scores = {
                GraspPhase.STABILIZE: 10,
                GraspPhase.APPROACH: 20,
                GraspPhase.CONTACT: 35,
                GraspPhase.GRASP: 55,
                GraspPhase.LIFT: 75,
                GraspPhase.HOLD: 100
            }
            
            score += phase_scores.get(self.current_phase, 0)
            
            # Bonus pour réussites spécifiques
            if self.successful_grasp:
                score += 20
            
            if self.cube_lifted:
                score += 25
            
            if self.hold_duration > 20:
                score += 15
            
            # Malus pour instabilité
            max_velocity = np.max(np.abs(self.data.qvel))
            if max_velocity > 5.0:
                score -= 10
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.error(f"❌ Erreur calcul score performance: {e}")
            return 0.0
    
    def _check_termination(self) -> bool:
        """Conditions de terminaison de l'épisode"""
        try:
            level_config = self.curriculum_levels[self.current_level]
            
            # Succès complet: cube soulevé et maintenu
            if (self.cube_lifted and 
                self.successful_grasp and 
                self.current_phase == GraspPhase.HOLD and
                self.hold_duration >= 30):
                
                self.logger.info("🎉 Succès complet: cube soulevé et maintenu!")
                return True
            
            # Échec critique: cube tombé loin
            cube_pos = self._get_cube_position()
            if (cube_pos[2] < -0.05 or  # Tombé sous la table
                abs(cube_pos[0]) > 2.0 or  # Trop loin en X
                abs(cube_pos[1]) > 2.0):   # Trop loin en Y
                
                self.logger.warning("⚠️ Cube perdu - terminaison")
                return True
            
            # Échec d'instabilité persistante
            if self.monitoring['max_velocity_violations'] > 50:
                self.logger.warning("⚠️ Instabilité excessive - terminaison")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Erreur vérification terminaison: {e}")
            return True
    
    def _update_monitoring(self, action, reward, observation):
        """Met à jour le système de monitoring"""
        try:
            # Enregistrer les métriques
            max_velocity = np.max(np.abs(self.data.qvel))
            self.monitoring['velocities'].append(float(max_velocity))
            self.monitoring['rewards'].append(float(reward))
            self.monitoring['phases'].append(self.current_phase.value)
            self.monitoring['contacts'].append(int(self._detect_robust_contact()))
            
            cube_pos = self._get_cube_position()
            self.monitoring['cube_positions'].append(cube_pos.tolist())
            
            # Limiter la taille des historiques
            max_history = 1000
            for key in ['velocities', 'rewards', 'phases', 'contacts', 'cube_positions']:
                if len(self.monitoring[key]) > max_history:
                    self.monitoring[key] = self.monitoring[key][-max_history:]
            
            # Logging périodique
            if self.episode_step % 100 == 0:
                avg_reward = np.mean(self.monitoring['rewards'][-50:]) if self.monitoring['rewards'] else 0
                self.logger.info(
                    f"Step {self.episode_step}: Phase={self.current_phase.name}, "
                    f"Level={self.current_level}, AvgReward={avg_reward:.2f}, "
                    f"MaxVel={max_velocity:.2f}, Contacts={sum(self.monitoring['contacts'][-10:])}"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Erreur monitoring: {e}")
    
    def render(self, mode='human'):
        """Rendu de l'environnement"""
        try:
            if mode == 'human':
                # Le viewer MuJoCo est géré par le thread séparé
                return None
            
            elif mode == 'rgb_array':
                # Rendu pour capture d'image
                width, height = 640, 480
                
                # Configuration de la caméra
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat = np.array([0.3, 0.0, 0.1])
                camera.distance = 1.0
                camera.azimuth = 45
                camera.elevation = -20
                
                # Créer le contexte de rendu
                if not hasattr(self, 'render_context'):
                    self.render_context = mujoco.MjrContext(
                        self.model, mujoco.mjtFontScale.mjFONTSCALE_150
                    )
                
                # Créer la scène
                scene = mujoco.MjvScene(self.model, maxgeom=10000)
                viewport = mujoco.MjrRect(0, 0, width, height)
                
                # Mettre à jour et rendre
                mujoco.mjv_updateScene(
                    self.model, self.data, mujoco.MjvOption(), 
                    None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                
                mujoco.mjr_render(viewport, scene, self.render_context)
                
                # Lire les pixels
                rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb_array, None, viewport, self.render_context)
                
                return np.flipud(rgb_array)
            
        except Exception as e:
            self.logger.error(f"❌ Erreur rendu: {e}")
            if mode == 'rgb_array':
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return None
    
    def close(self):
        """Nettoyage des ressources"""
        try:
            # Arrêter le thread du viewer
            if hasattr(self, 'viewer_thread'):
                # Le thread daemon se fermera automatiquement
                pass
            
            # Nettoyer le fichier temporaire
            if hasattr(self, 'temp_model_path') and os.path.exists(self.temp_model_path):
                try:
                    os.unlink(self.temp_model_path)
                    self.logger.info("🗑️ Fichier temporaire nettoyé")
                except:
                    pass
            
            # Sauvegarder les statistiques de monitoring
            if hasattr(self, 'monitoring'):
                self._save_monitoring_stats()
            
            self.logger.info("🏁 Environnement fermé proprement")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur fermeture: {e}")
    
    def _save_monitoring_stats(self):
        """Sauvegarde les statistiques de monitoring"""
        try:
            import json
            
            stats_dir = "/tmp/grasp_env_stats"
            os.makedirs(stats_dir, exist_ok=True)
            
            timestamp = int(time.time())
            stats_file = os.path.join(stats_dir, f"stats_{timestamp}.json")
            
            # Préparer les données pour JSON
            json_data = {
                'curriculum_level': self.current_level,
                'total_episodes': len(self.performance_history),
                'consecutive_successes': self.consecutive_successes,
                'performance_history': self.performance_history,
                'monitoring_summary': {
                    'avg_velocity': float(np.mean(self.monitoring['velocities'])) 
                                  if self.monitoring['velocities'] else 0.0,
                    'avg_reward': float(np.mean(self.monitoring['rewards'])) 
                                if self.monitoring['rewards'] else 0.0,
                    'total_contacts': int(sum(self.monitoring['contacts'])),
                    'velocity_violations': self.monitoring['max_velocity_violations'],
                    'stability_violations': self.monitoring['stability_violations']
                }
            }
            
            with open(stats_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"📊 Statistiques sauvegardées: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde stats: {e}")
    
    def get_curriculum_status(self) -> Dict:
        """Retourne le statut détaillé du curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        
        return {
            'current_level': self.current_level,
            'level_name': level_config.name,
            'level_description': level_config.description,
            'progress': {
                'consecutive_successes': self.consecutive_successes,
                'required_successes': level_config.episodes_required,
                'current_threshold': level_config.success_threshold,
                'episodes_in_level': self.level_episodes
            },
            'phase_status': {
                'current_phase': self.current_phase.name,
                'max_phases': level_config.max_phases,
                'phase_timer': self.phase_timer,
                'phase_success_count': dict(self.phase_success_count)
            },
            'performance': {
                'recent_scores': self.performance_history[-5:],
                'average_score': np.mean(self.performance_history) if self.performance_history else 0.0,
                'best_score': max(self.performance_history) if self.performance_history else 0.0
            }
        }
    
    def advance_curriculum_level(self, episode_reward: float):
        """Fait avancer le curriculum si les conditions sont remplies"""
        try:
            # Ajouter la performance
            self.performance_history.append(episode_reward)
            self.level_episodes += 1
            
            level_config = self.curriculum_levels[self.current_level]
            
            # Vérifier si succès
            if episode_reward >= level_config.success_threshold:
                self.consecutive_successes += 1
                self.logger.info(f"✅ Succès niveau {self.current_level}: {self.consecutive_successes}/{level_config.episodes_required}")
            else:
                self.consecutive_successes = 0
            
            # Vérifier si on peut passer au niveau suivant
            if (self.consecutive_successes >= level_config.episodes_required and
                self.current_level < len(self.curriculum_levels)):
                
                old_level = self.current_level
                self.current_level += 1
                self.consecutive_successes = 0
                self.level_episodes = 0
                self.level_start_time = time.time()
                
                # Mettre à jour la configuration des phases
                self._update_phase_config()
                
                # Log de la transition
                new_config = self.curriculum_levels[self.current_level]
                self.logger.info("🎓" + "="*60)
                self.logger.info(f"🎓 PASSAGE AU NIVEAU {self.current_level}!")
                self.logger.info(f"🎓 {old_level}: {level_config.name} → {self.current_level}: {new_config.name}")
                self.logger.info(f"🎓 Description: {new_config.description}")
                self.logger.info(f"🎓 Phases disponibles: {new_config.max_phases}")
                self.logger.info(f"🎓 Objectif: {new_config.success_threshold:.1f} points")
                self.logger.info("🎓" + "="*60)
                
                # Enregistrer la transition
                self.monitoring['curriculum_transitions'].append({
                    'timestamp': time.time(),
                    'from_level': old_level,
                    'to_level': self.current_level,
                    'episode_reward': episode_reward,
                    'total_episodes': len(self.performance_history)
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Erreur advancement curriculum: {e}")
            return False


# Fonction utilitaire pour créer l'environnement
def make_ultra_robust_grasp_env(**kwargs):
    """
    Factory function pour créer l'environnement de grasping ultra-robuste
    
    Args:
        **kwargs: Arguments à passer au constructeur de l'environnement
    
    Returns:
        UltraRobustGraspEnv: Instance de l'environnement configuré
    """
    try:
        env = UltraRobustGraspEnv(**kwargs)
        return env
    except Exception as e:
        print(f"❌ Erreur création environnement: {e}")
        raise


# Test rapide de l'environnement
if __name__ == "__main__":
    import sys
    
    print("🎯 Test de l'Environnement de Grasping Ultra-Robuste")
    print("=" * 60)
    
    try:
        # Créer l'environnement
        env = make_ultra_robust_grasp_env(
            render_mode="human",
            enable_curriculum=True,
            enable_mujoco_viewer=True
        )
        
        print(f"✅ Environnement créé")
        print(f"📊 Espace d'action: {env.action_space.shape}")
        print(f"📊 Espace d'observation: {env.observation_space.shape}")
        
        # Test de quelques épisodes
        for episode in range(3):
            print(f"\n🔄 Épisode {episode + 1}")
            
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(200):
                # Action aléatoire
                action = env.action_space.sample() * 0.1  # Actions douces
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"📊 Épisode {episode + 1}: {steps} steps, reward={total_reward:.2f}")
            print(f"📈 Phase finale: {info.get('phase', 'Unknown')}")
            print(f"🎓 Niveau curriculum: {info.get('curriculum_level', 'Unknown')}")
            
            # Faire avancer le curriculum avec la récompense obtenue
            env.advance_curriculum_level(total_reward)
        
        # Afficher le statut du curriculum
        status = env.get_curriculum_status()
        print(f"\n🎓 Statut Curriculum Final:")
        print(f"   Niveau: {status['current_level']} - {status['level_name']}")
        print(f"   Progrès: {status['progress']['consecutive_successes']}/{status['progress']['required_successes']}")
        print(f"   Score moyen: {status['performance']['average_score']:.2f}")
        
        env.close()
        print("\n✅ Test terminé avec succès!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu par l'utilisateur")
        if 'env' in locals():
            env.close()
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()
        sys.exit(1)