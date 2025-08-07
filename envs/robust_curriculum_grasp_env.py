#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT DE GRASPING ROBUSTE AVEC CURRICULUM LEARNING
============================================================
Version ultra-stable et professionnelle qui corrige tous les problèmes:
✅ Vitesses excessives - Contrôle de vitesse intelligent
✅ Erreurs mujoco - Gestion robuste des imports et contextes
✅ Capture vidéo - Système de vidéo intégré et fonctionnel
✅ Stagnation - Système de récompenses adaptatif
✅ Instabilité - Physique ultra-stable
✅ Monitoring - Suivi en temps réel des performances
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
import json
import tempfile
import warnings
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
warnings.filterwarnings("ignore")

# Import mujoco avec gestion d'erreurs robuste
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Mujoco non disponible: {e}")
    MUJOCO_AVAILABLE = False

class RobustCurriculumGraspEnv(gym.Env):
    """
    🎯 Environnement de Grasping Ultra-Robuste avec Curriculum Learning
    
    Fonctionnalités:
    - Contrôle de vitesse intelligent pour éviter les vitesses excessives
    - Gestion robuste des erreurs mujoco
    - Capture vidéo intégrée et fonctionnelle
    - Système de récompenses adaptatif
    - Physique ultra-stable
    - Monitoring en temps réel
    """
    
    def __init__(self, model_path: str = None, render_mode: str = None, video_capture: bool = True):
        super().__init__()
        
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("Mujoco n'est pas disponible. Installez mujoco-py ou mujoco.")
        
        # Configuration
        self.render_mode = render_mode
        self.video_capture = video_capture
        self.model_path_str = model_path or "/workspace/results/g1_combined.xml"
        
        # Configuration du curriculum learning
        self.curriculum_levels = {
            1: {
                'name': 'STABILIZATION_ONLY',
                'description': 'Apprendre à stabiliser les bras',
                'max_phases': 1,
                'success_threshold': 15.0,
                'episodes_required': 5,
                'max_episode_steps': 200,
                'cube_fixed': True,
                'reward_multiplier': 1.0,
                'max_velocity': 2.0,
                'action_scale': 0.1
            },
            2: {
                'name': 'APPROACH_LEARNING',
                'description': 'Apprendre à approcher le cube',
                'max_phases': 2,
                'success_threshold': 25.0,
                'episodes_required': 5,
                'max_episode_steps': 300,
                'cube_fixed': True,
                'reward_multiplier': 1.2,
                'max_velocity': 3.0,
                'action_scale': 0.15
            },
            3: {
                'name': 'CONTACT_LEARNING',
                'description': 'Apprendre à toucher le cube',
                'max_phases': 3,
                'success_threshold': 40.0,
                'episodes_required': 4,
                'max_episode_steps': 400,
                'cube_fixed': False,
                'reward_multiplier': 1.5,
                'max_velocity': 4.0,
                'action_scale': 0.2
            },
            4: {
                'name': 'FULL_GRASPING',
                'description': 'Grasping complet',
                'max_phases': 6,
                'success_threshold': 60.0,
                'episodes_required': 3,
                'max_episode_steps': 500,
                'cube_fixed': False,
                'reward_multiplier': 2.0,
                'max_velocity': 5.0,
                'action_scale': 0.25
            },
            5: {
                'name': 'MASTERY',
                'description': 'Maîtrise complète',
                'max_phases': 6,
                'success_threshold': 80.0,
                'episodes_required': 2,
                'max_episode_steps': 600,
                'cube_fixed': False,
                'reward_multiplier': 2.5,
                'max_velocity': 6.0,
                'action_scale': 0.3
            }
        }
        
        # Initialisation des variables
        self.current_level = 1
        self.current_phase = 0
        self.phase_timer = 0
        self.episode_step = 0
        self.stability_count = 0
        self.successful_grasp = False
        self.cube_lifted = False
        self.consecutive_successes = 0
        self.level_episodes = 0
        self.performance_history = []
        
        # Configuration de la vidéo
        self.video_writer = None
        self.video_path = None
        self.frame_count = 0
        
        # Initialisation du modèle et de la physique
        self._setup_model()
        self._identify_components()
        self._setup_spaces()
        
        if self.video_capture:
            self._setup_video_capture()
        
        print(f"🎯 RobustCurriculumGraspEnv initialisé - Niveau {self.current_level}")
        print(f"📁 Modèle: {self.model_path_str}")
        print(f"🎥 Capture vidéo: {self.video_capture}")

    def _setup_model(self):
        """Configure le modèle avec physique ultra-stable"""
        try:
            # Charger le modèle
            if not os.path.exists(self.model_path_str):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path_str}")
            
            # Lire le contenu XML
            with open(self.model_path_str, 'r') as f:
                xml_content = f.read()
            
            # Appliquer les corrections de physique
            xml_content = self._apply_ultra_physics_fixes(xml_content)
            
            # Corriger les chemins relatifs pour les assets
            xml_content = self._fix_asset_paths(xml_content)
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                self.temp_model_path = f.name
            
            # Charger le modèle
            self.model = mujoco.MjModel.from_xml_path(self.temp_model_path)
            self.data = mujoco.MjData(self.model)
            
            # Configuration ultra-stable
            self.model.opt.timestep = 0.0005  # Timestep très petit
            self.model.opt.iterations = 50    # Plus d'itérations
            self.model.opt.tolerance = 1e-8   # Tolérance très stricte
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # Solver CG plus stable
            
            print(f"✅ Modèle chargé avec physique ultra-stable - DOFs: {self.model.nv} - Actuateurs: {self.model.nu} - Timestep: {self.model.opt.timestep}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def _apply_ultra_physics_fixes(self, xml_content: str) -> str:
        """Applique des corrections de physique ultra-stables"""
        # Réduire les frottements
        xml_content = xml_content.replace('<friction>0.8</friction>', '<friction>0.3</friction>')
        xml_content = xml_content.replace('<friction>0.6</friction>', '<friction>0.2</friction>')
        
        # Augmenter le damping
        xml_content = xml_content.replace('<damping>0.1</damping>', '<damping>0.5</damping>')
        xml_content = xml_content.replace('<damping>0.05</damping>', '<damping>0.3</damping>')
        
        # Réduire les masses pour plus de stabilité
        xml_content = xml_content.replace('<mass>1.0</mass>', '<mass>0.5</mass>')
        xml_content = xml_content.replace('<mass>0.5</mass>', '<mass>0.3</mass>')
        
        return xml_content

    def _fix_asset_paths(self, xml_content: str) -> str:
        """Corrige les chemins relatifs des assets"""
        # Remplacer les chemins relatifs par des chemins absolus
        base_dir = "/workspace"
        
        # Remplacer les includes
        xml_content = xml_content.replace(
            'file="../assets/hands/g1_body.xml"',
            f'file="{base_dir}/assets/hands/g1_body.xml"'
        )
        xml_content = xml_content.replace(
            'file="../assets/hands/g1_fingers.xml"',
            f'file="{base_dir}/assets/hands/g1_fingers.xml"'
        )
        
        return xml_content

    def _identify_components(self):
        """Identifie les composants du robot"""
        # Identifier les joints des bras (premiers joints)
        self.arm_joint_ids = list(range(min(14, self.model.njnt)))
        
        # Identifier les joints des doigts (recherche par nom)
        self.finger_joint_ids = []
        finger_names = ['finger', 'digit', 'phalange', 'joint']
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_name_lower = joint_name.lower()
                if any(name in joint_name_lower for name in finger_names):
                    self.finger_joint_ids.append(i)
        
        # Si aucun joint de doigt trouvé, utiliser des indices par défaut
        if not self.finger_joint_ids:
            print("⚠️ Aucun joint de doigt trouvé, utilisation de tous les joints restants comme doigts")
            total_joints = self.model.njnt
            self.finger_joint_ids = list(range(14, total_joints))
        
        # Identifier le cube
        self.cube_body_id = -1
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and ('cube' in body_name.lower() or 'object' in body_name.lower()):
                self.cube_body_id = i
                break
        
        if self.cube_body_id == -1:
            # Utiliser le dernier body comme cube
            self.cube_body_id = self.model.nbody - 1
        
        print(f"✅ Composants identifiés:")
        print(f"   - Joints bras: {len(self.arm_joint_ids)}")
        print(f"   - Joints doigts: {len(self.finger_joint_ids)}")
        print(f"   - Cube ID: {self.cube_body_id}")

    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        # Espace d'observation: positions, vitesses, positions du cube et de la main
        obs_dim = (len(self.arm_joint_ids) * 2 +  # positions et vitesses des bras
                  len(self.finger_joint_ids) * 2 +  # positions et vitesses des doigts
                  3 +  # position du cube
                  3 +  # position de la main
                  6 +  # vitesse du cube
                  3)   # phase, timer, niveau
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Espace d'action: contrôle des joints
        action_dim = len(self.arm_joint_ids) + len(self.finger_joint_ids)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def _setup_video_capture(self):
        """Configure la capture vidéo"""
        if not self.video_capture:
            return
        
        try:
            # Créer le dossier vidéo
            video_dir = "/workspace/results/videos"
            os.makedirs(video_dir, exist_ok=True)
            
            # Nom du fichier vidéo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(video_dir, f"grasp_training_{timestamp}.mp4")
            
            # Configuration du writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (640, 480))
            
            if self.video_writer.isOpened():
                print(f"🎥 Capture vidéo configurée: {self.video_path}")
            else:
                print("⚠️ Impossible d'ouvrir le writer vidéo")
                self.video_writer = None
                
        except Exception as e:
            print(f"⚠️ Erreur configuration vidéo: {e}")
            self.video_writer = None

    def update_curriculum_level(self, episode_reward: float, episode_success: bool):
        """Met à jour le niveau de curriculum"""
        self.performance_history.append(episode_reward)
        self.level_episodes += 1
        
        # Vérifier si on peut passer au niveau suivant
        if episode_success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        
        current_config = self.curriculum_levels[self.current_level]
        
        if (self.consecutive_successes >= current_config['episodes_required'] and 
            episode_reward >= current_config['success_threshold'] and
            self.current_level < len(self.curriculum_levels)):
            
            self.current_level += 1
            self.consecutive_successes = 0
            self.level_episodes = 0
            print(f"🎓 Passage au niveau {self.current_level}: {self.curriculum_levels[self.current_level]['name']}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset des données mujoco
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset des variables
        self.current_phase = 0
        self.phase_timer = 0
        self.episode_step = 0
        self.stability_count = 0
        self.successful_grasp = False
        self.cube_lifted = False
        
        # Position initiale du cube
        cube_pos = [0.4, 0.0, 0.02]
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.qpos):
            cube_start_idx = self.cube_body_id * 7
            if cube_start_idx + 3 <= len(self.data.qpos):
                self.data.qpos[cube_start_idx:cube_start_idx+3] = cube_pos
        
        # Reset des contrôles
        if hasattr(self.data, 'ctrl') and self.data.ctrl is not None:
            self.data.ctrl[:] = 0.0
        
        # Première observation
        obs = self._get_observation()
        # --- Correction SB3 : forcer float32 et shape ---
        obs = np.array(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.observation_space.shape[0]:
            raise RuntimeError(f"Observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}")
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        try:
            self.episode_step += 1
            self.phase_timer += 1
            scaled_action = self._apply_curriculum_scaling(action)
            self._apply_smooth_actions(scaled_action)
            mujoco.mj_step(self.model, self.data)
            self._check_stability()
            self._update_phase_curriculum()
            reward = self._calculate_curriculum_reward()
            terminated = self._check_termination()
            truncated = self.episode_step >= self.curriculum_levels[self.current_level]['max_episode_steps']
            obs = self._get_observation()
            obs = np.array(obs, dtype=np.float32).reshape(-1)
            if obs.shape[0] != self.observation_space.shape[0]:
                raise RuntimeError(f"Observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}")
            info = self._get_info()
            if self.video_capture:
                self._capture_frame()
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"🚨 Erreur critique Mujoco/physique : {e} - redémarrage automatique de l'épisode !")
            obs, info = self.reset()
            return obs, 0.0, True, True, info

    def _apply_curriculum_scaling(self, action):
        """Applique le scaling du curriculum à l'action"""
        current_config = self.curriculum_levels[self.current_level]
        action_scale = current_config['action_scale']
        
        # Scaling adaptatif basé sur le niveau
        scaled_action = action * action_scale
        
        # Limiter les actions extrêmes
        scaled_action = np.clip(scaled_action, -1.0, 1.0)
        
        return scaled_action

    def _apply_smooth_actions(self, action):
        try:
            arm_actions = action[:len(self.arm_joint_ids)]
            finger_actions = action[len(self.arm_joint_ids):len(self.arm_joint_ids) + len(self.finger_joint_ids)]
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(arm_actions) and i < len(self.data.ctrl):
                    current_pos = float(self.data.qpos[joint_id])
                    current_vel = float(self.data.qvel[joint_id])
                    # Clipping strict de l'action
                    act = float(np.clip(arm_actions[i], -0.5, 0.5))
                    # Scaling adaptatif
                    action_scale = 0.05 if self.current_level <= 2 else 0.1
                    # Si vitesse > 8, scaling réduit
                    if abs(current_vel) > 8.0:
                        action_scale *= 0.5
                    target_pos = current_pos + act * action_scale
                    max_change = 0.02 if self.current_level <= 2 else 0.03
                    target_pos = np.clip(target_pos, current_pos - max_change, current_pos + max_change)
                    # Appliquer avec contrôle de vitesse
                    if abs(current_vel) > 10.0:
                        if not hasattr(self, '_vel_excess_count'):
                            self._vel_excess_count = {}
                        self._vel_excess_count[joint_id] = self._vel_excess_count.get(joint_id, 0) + 1
                        if self._vel_excess_count[joint_id] > 10:
                            print(f"🚨 Vitesse persistante >10 sur joint {joint_id} - forçage qvel=0 !")
                            self.data.qvel[joint_id] = 0.0
                            self._vel_excess_count[joint_id] = 0
                        target_pos = current_pos  # Ne pas bouger
                    else:
                        if hasattr(self, '_vel_excess_count'):
                            self._vel_excess_count[joint_id] = 0
                    self.data.ctrl[i] = target_pos
            for i, joint_id in enumerate(self.finger_joint_ids):
                if i < len(finger_actions) and (14 + i) < len(self.data.ctrl):
                    current_pos = float(self.data.qpos[joint_id])
                    current_vel = float(self.data.qvel[joint_id])
                    act = float(np.clip(finger_actions[i], -0.5, 0.5))
                    action_scale = 0.1
                    if abs(current_vel) > 8.0:
                        action_scale *= 0.5
                    target_pos = current_pos + act * action_scale
                    if abs(current_vel) > 10.0:
                        if not hasattr(self, '_vel_excess_count_f'): self._vel_excess_count_f = {}
                        self._vel_excess_count_f[joint_id] = self._vel_excess_count_f.get(joint_id, 0) + 1
                        if self._vel_excess_count_f[joint_id] > 10:
                            print(f"🚨 Vitesse persistante >10 sur doigt {joint_id} - forçage qvel=0 !")
                            self.data.qvel[joint_id] = 0.0
                            self._vel_excess_count_f[joint_id] = 0
                        target_pos = current_pos
                    else:
                        if hasattr(self, '_vel_excess_count_f'):
                            self._vel_excess_count_f[joint_id] = 0
                    self.data.ctrl[14 + i] = target_pos
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur application actions: {e}")

    def _check_stability(self):
        """Vérifie la stabilité du système et réduit automatiquement les vitesses excessives."""
        try:
            # Correction automatique NaN/Inf sur qpos, qvel, ctrl
            for arr_name in ['qpos', 'qvel', 'ctrl']:
                arr = getattr(self.data, arr_name, None)
                if arr is not None and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
                    print(f"🚨 Correction automatique : {arr_name} contenait NaN/Inf, réinitialisation à zéro !")
                    arr[...] = 0.0
            # Vérifier les vitesses des joints bras et doigts
            max_velocity = 0.0
            excessive_count = 0
            for joint_id in self.arm_joint_ids + self.finger_joint_ids:
                if joint_id < len(self.data.qvel):
                    velocity = abs(float(self.data.qvel[joint_id]))
                    max_velocity = max(max_velocity, velocity)
                    if velocity > 20.0:
                        excessive_count += 1
            # Si vitesse critique, réduire toutes les vitesses
            if max_velocity > 20.0:
                print(f"🚨 Vitesse critique détectée ({max_velocity:.2f}) - réduction drastique appliquée à tous les joints !")
                for i in range(len(self.data.qvel)):
                    self.data.qvel[i] *= 0.1
                self.stability_count = 0
            # Si vitesse excessive, réduire bras et doigts séparément
            elif max_velocity > 10.0:
                print(f"⚠️ Vitesse excessive ({max_velocity:.2f}) - réduction appliquée")
                for joint_id in self.arm_joint_ids:
                    if joint_id < len(self.data.qvel):
                        self.data.qvel[joint_id] *= 0.3
                for joint_id in self.finger_joint_ids:
                    if joint_id < len(self.data.qvel):
                        self.data.qvel[joint_id] *= 0.2
                self.stability_count = 0
            # Clipping strict sur toutes les vitesses
            self.data.qvel = np.clip(self.data.qvel, -25.0, 25.0)
            # Si la vitesse reste excessive trop longtemps, stopper l'épisode
            if max_velocity > 10.0:
                self.stability_count += 1
                if self.stability_count > 20:
                    print("🚨 Instabilité persistante : terminaison anticipée de l'épisode !")
                    self.episode_step = self.curriculum_levels[self.current_level]['max_episode_steps']
            else:
                self.stability_count = 0
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur vérification stabilité: {e}")

    def _update_phase_curriculum(self):
        """Met à jour la phase du curriculum"""
        current_config = self.curriculum_levels[self.current_level]
        max_phases = current_config['max_phases']
        
        # Transition de phase basée sur le temps et les performances
        if self.phase_timer > 50 and self.current_phase < max_phases - 1:
            # Vérifier si on peut passer à la phase suivante
            cube_pos = self._get_cube_position()
            hand_pos = self._get_hand_center()
            
            distance = np.linalg.norm(cube_pos - hand_pos)
            
            if distance < 0.1 and self.current_phase == 0:  # APPROACH
                self.current_phase = 1
                self.phase_timer = 0
            elif distance < 0.05 and self.current_phase == 1:  # CONTACT
                self.current_phase = 2
                self.phase_timer = 0
            elif self._detect_finger_contact() and self.current_phase == 2:  # GRASP
                self.current_phase = 3
                self.phase_timer = 0
            elif self._is_cube_lifted() and self.current_phase == 3:  # LIFT
                self.current_phase = 4
                self.phase_timer = 0
            elif self._check_grasp_stability() and self.current_phase == 4:  # HOLD
                self.current_phase = 5
                self.phase_timer = 0

    def _calculate_curriculum_reward(self):
        """Calcule la récompense adaptative"""
        reward = 0.0
        
        try:
            # Récompense de base pour la stabilité
            if self.stability_count > 10:
                reward += 1.0
            
            # Récompense basée sur la phase
            if self.current_phase == 0:  # STABILIZE
                # Récompense pour la stabilité des bras
                arm_velocities = [abs(float(self.data.qvel[i])) for i in self.arm_joint_ids if i < len(self.data.qvel)]
                if arm_velocities:
                    avg_velocity = np.mean(arm_velocities)
                    if avg_velocity < 0.5:
                        reward += 2.0
                    elif avg_velocity < 1.0:
                        reward += 1.0
            
            elif self.current_phase == 1:  # APPROACH
                # Récompense pour s'approcher du cube
                cube_pos = self._get_cube_position()
                hand_pos = self._get_hand_center()
                distance = np.linalg.norm(cube_pos - hand_pos)
                
                if distance < 0.2:
                    reward += 3.0
                elif distance < 0.3:
                    reward += 1.5
                elif distance < 0.5:
                    reward += 0.5
            
            elif self.current_phase == 2:  # CONTACT
                # Récompense pour toucher le cube
                if self._detect_finger_contact():
                    reward += 5.0
            
            elif self.current_phase == 3:  # GRASP
                # Récompense pour saisir le cube
                if self._check_grasp_stability():
                    reward += 8.0
            
            elif self.current_phase == 4:  # LIFT
                # Récompense pour soulever le cube
                if self._is_cube_lifted():
                    reward += 10.0
            
            elif self.current_phase == 5:  # HOLD
                # Récompense pour maintenir le cube
                if self._check_grasp_stability() and self._is_cube_lifted():
                    reward += 15.0
            
            # Pénalité pour les vitesses excessives
            max_velocity = 0.0
            for joint_id in self.arm_joint_ids:
                if joint_id < len(self.data.qvel):
                    velocity = abs(float(self.data.qvel[joint_id]))
                    max_velocity = max(max_velocity, velocity)
            
            if max_velocity > 5.0:
                reward -= 2.0
            
            # Pénalité forte pour les vitesses excessives
            max_velocity = 0.0
            for joint_id in self.arm_joint_ids + self.finger_joint_ids:
                if joint_id < len(self.data.qvel):
                    velocity = abs(float(self.data.qvel[joint_id]))
                    max_velocity = max(max_velocity, velocity)
            if max_velocity > 10.0:
                reward -= 10.0  # pénalité forte
            
            # Multiplicateur du curriculum
            current_config = self.curriculum_levels[self.current_level]
            reward *= current_config['reward_multiplier']
            
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur calcul récompense: {e}")
            reward = 0.0
        
        return reward

    def _get_observation(self):
        """Retourne l'observation actuelle avec types de données cohérents"""
        obs = []
        
        try:
            # Position et vitesse des bras
            for joint_id in self.arm_joint_ids:
                if joint_id < len(self.data.qpos):
                    obs.append(float(self.data.qpos[joint_id]))
                else:
                    obs.append(0.0)
                if joint_id < len(self.data.qvel):
                    obs.append(float(self.data.qvel[joint_id]))
                else:
                    obs.append(0.0)
            
            # Position et vitesse des doigts
            for joint_id in self.finger_joint_ids:
                if joint_id < len(self.data.qpos):
                    obs.append(float(self.data.qpos[joint_id]))
                else:
                    obs.append(0.0)
                if joint_id < len(self.data.qvel):
                    obs.append(float(self.data.qvel[joint_id]))
                else:
                    obs.append(0.0)
            
            # Position du cube
            cube_pos = self._get_cube_position()
            obs.extend([float(x) for x in cube_pos])
            
            # Position de la main
            hand_pos = self._get_hand_center()
            obs.extend([float(x) for x in hand_pos])
            
            # Vitesse du cube
            if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.cvel):
                cube_vel = self.data.cvel[self.cube_body_id]
                obs.extend([float(cube_vel[0]), float(cube_vel[1]), float(cube_vel[2]), 
                           float(cube_vel[3]), float(cube_vel[4]), float(cube_vel[5])])
            else:
                obs.extend([0.0] * 6)
            
            # Phase et timer
            obs.append(float(self.current_phase))
            obs.append(float(self.phase_timer))
            
            # Niveau de curriculum
            obs.append(float(self.current_level))
            
            # Convertir en array numpy avec type float32
            obs_array = np.array(obs, dtype=np.float32)
            
            # --- Robustesse : gestion NaN/Inf ---
            if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
                print("⚠️ Observation contient NaN/Inf - remplacement par zéros")
                obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
            # Robustesse : clip valeurs aberrantes
            if np.any(obs_array > 1e6) or np.any(obs_array < -1e6):
                print("⚠️ Observation contient des valeurs aberrantes - clipping à [-1e6, 1e6]")
                obs_array = np.clip(obs_array, -1e6, 1e6)
            
            return obs_array
            
        except Exception as e:
            print(f"⚠️ Erreur observation: {e}")
            # Retourner une observation par défaut
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32)

    def _get_info(self):
        """Retourne les informations de l'environnement"""
        return {
            'phase': self._get_phase_name(),
            'curriculum_level': self.current_level,
            'episode_step': self.episode_step,
            'stability_count': self.stability_count,
            'successful_grasp': self.successful_grasp,
            'cube_lifted': self.cube_lifted,
            'avg_velocity': np.mean([abs(self.data.qvel[i]) for i in self.arm_joint_ids if i < len(self.data.qvel)])
        }

    def _check_termination(self):
        """Vérifie si l'épisode doit se terminer"""
        # Terminer si cube soulevé et tenu stable
        if self.cube_lifted and self._check_grasp_stability():
            return True
        
        # Terminer si trop d'étapes
        if self.episode_step >= self.curriculum_levels[self.current_level]['max_episode_steps']:
            return True
        
        return False

    def _get_phase_name(self):
        """Retourne le nom de la phase actuelle"""
        phases = ['STABILIZE', 'APPROACH', 'CONTACT', 'GRASP', 'LIFT', 'HOLD']
        return phases[min(self.current_phase, len(phases) - 1)]

    def _get_cube_position(self):
        """Retourne la position du cube"""
        try:
            if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
                pos = self.data.xpos[self.cube_body_id]
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur position cube: {e}")
        return np.array([0.4, 0.0, 0.02], dtype=np.float32)

    def _get_hand_center(self):
        """Retourne la position centrale de la main"""
        try:
            # Calculer la position moyenne des doigts
            finger_positions = []
            for joint_id in self.finger_joint_ids:
                if joint_id < len(self.data.xpos):
                    pos = self.data.xpos[joint_id]
                    finger_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
            
            if finger_positions:
                mean_pos = np.mean(finger_positions, axis=0)
                return np.array([float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])], dtype=np.float32)
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur position main: {e}")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _detect_finger_contact(self):
        """Détecte le contact avec les doigts"""
        try:
            if self.cube_body_id < 0:
                return False
            
            # Vérifier les contacts
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if (contact.geom1 == self.cube_body_id or contact.geom2 == self.cube_body_id):
                    return True
            
            return False
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur détection contact: {e}")
            return False

    def _check_grasp_stability(self):
        """Vérifie la stabilité de la saisie"""
        try:
            if not self._detect_finger_contact():
                return False
            
            # Vérifier si le cube est maintenu fermement
            cube_pos = self._get_cube_position()
            hand_pos = self._get_hand_center()
            distance = np.linalg.norm(cube_pos - hand_pos)
            
            return distance < 0.1
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur stabilité saisie: {e}")
            return False

    def _is_cube_lifted(self):
        """Vérifie si le cube est soulevé"""
        try:
            cube_pos = self._get_cube_position()
            return cube_pos[2] > 0.05  # Plus de 5cm au-dessus de la table
        except Exception as e:
            if self.episode_step % 100 == 0:
                print(f"⚠️ Erreur vérification soulevé: {e}")
            return False

    def _capture_frame(self):
        """Capture une frame pour la vidéo"""
        try:
            if self.video_writer is not None and self.video_writer.isOpened():
                # Rendu de la frame
                frame = self.render()
                
                if frame is not None and frame.size > 0:
                    # Convertir BGR pour OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Écrire la frame
                    self.video_writer.write(frame_bgr)
                    self.frame_count += 1
                
        except Exception as e:
            if hasattr(self, 'episode_step') and self.episode_step % 100 == 0:
                print(f"⚠️ Erreur capture frame: {e}")

    def render(self):
        """Rendu de l'environnement avec gestion robuste des erreurs"""
        try:
            if self.render_mode == "human":
                try:
                    # Créer un viewer si nécessaire
                    if not hasattr(self, 'viewer') or self.viewer is None:
                        import mujoco.viewer
                        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    
                    if hasattr(self, 'viewer') and self.viewer is not None:
                        self.viewer.sync()
                        
                except Exception as e:
                    print(f"⚠️ Erreur rendu human: {e}")
                    
            elif self.render_mode == "rgb_array":
                try:
                    # Configuration de la caméra
                    width, height = 640, 480
                    
                    # Créer un contexte de rendu si nécessaire
                    if not hasattr(self, 'render_context'):
                        self.render_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
                    
                    # Configuration de la scène
                    scene = mujoco.MjvScene(self.model, maxgeom=10000)
                    camera = mujoco.MjvCamera()
                    
                    # Position optimale de la caméra pour voir le grasping
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat = np.array([0.4, 0.0, 0.05])  # Regarder vers le cube
                    camera.distance = 1.2
                    camera.azimuth = 45
                    camera.elevation = -25
                    
                    # Mettre à jour la scène
                    mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    
                    # Créer le viewport
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    
                    # Rendu de la scène
                    mujoco.mjr_render(viewport, scene, self.render_context)
                    
                    # Lire les pixels
                    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb_array, None, viewport, self.render_context)
                    
                    # Retourner l'image (flip vertical car OpenGL utilise origine en bas)
                    return np.flipud(rgb_array)
                    
                except Exception as e:
                    if hasattr(self, 'episode_step') and self.episode_step % 100 == 0:
                        print(f"⚠️ Erreur rendu rgb_array: {e}")
                    # Retourner une image noire en cas d'erreur
                    return np.zeros((480, 640, 3), dtype=np.uint8)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur générale rendu: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """Nettoie les ressources"""
        # Fermer la vidéo
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"🎥 Vidéo sauvegardée: {self.video_path}")
        
        # Nettoyer le fichier temporaire
        if hasattr(self, 'temp_model_path') and os.path.exists(self.temp_model_path):
            try:
                os.unlink(self.temp_model_path)
            except:
                pass

    def get_curriculum_info(self):
        """Retourne les informations détaillées sur le curriculum"""
        return {
            'current_level': self.current_level,
            'level_name': self.curriculum_levels[self.current_level]['name'],
            'level_description': self.curriculum_levels[self.current_level]['description'],
            'consecutive_successes': self.consecutive_successes,
            'required_successes': self.curriculum_levels[self.current_level]['episodes_required'],
            'success_threshold': self.curriculum_levels[self.current_level]['success_threshold'],
            'level_episodes': self.level_episodes,
            'max_phases': self.curriculum_levels[self.current_level]['max_phases'],
            'performance_history': self.performance_history[-10:]
        }