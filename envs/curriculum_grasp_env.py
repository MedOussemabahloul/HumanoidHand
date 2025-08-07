#!/usr/bin/env python3
"""
🎓 ENVIRONNEMENT DE GRASPING AVEC CURRICULUM LEARNING
====================================================
Curriculum Learning intelligent qui adapte automatiquement la difficulté:
🎯 Niveau 1: Stabilisation des bras uniquement
🎯 Niveau 2: Stabilisation + Approche du cube  
🎯 Niveau 3: Stabilisation + Approche + Contact
🎯 Niveau 4: Grasping complet (toutes les phases)
🎯 Niveau 5: Grasping avec perturbations aléatoires
Progression automatique basée sur les performances !
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
import time
warnings.filterwarnings("ignore")

class CurriculumGraspEnv(gym.Env):
    """
    🎓 Environnement de Grasping avec Curriculum Learning Adaptatif
    
    Le curriculum learning adapte automatiquement la difficulté selon les performances:
    - Commence simple (stabilisation uniquement)
    - Progresse graduellement vers le grasping complet
    - Ajuste la difficulté en temps réel
    """
    
    def __init__(self, model_path: str = None, render_mode: str = None):
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.model_path_str = model_path or "/home/oussema/Documents/project/results/g1_combined.xml"
        
        # Curriculum Learning Configuration
        self.curriculum_levels = {
            1: {
                'name': 'STABILIZATION_ONLY',
                'description': 'Apprendre à stabiliser les bras',
                'max_phases': 1,  # Seulement STABILIZE
                'success_threshold': 15.0,  # Récompense requise pour passer au niveau suivant
                'episodes_required': 5,     # Nombre d'épisodes réussis consécutifs
                'max_episode_steps': 200,
                'cube_fixed': True,         # Cube ne bouge pas
                'reward_multiplier': 1.0
            },
            2: {
                'name': 'APPROACH_LEARNING',
                'description': 'Apprendre à approcher le cube',
                'max_phases': 2,  # STABILIZE + APPROACH
                'success_threshold': 25.0,
                'episodes_required': 5,
                'max_episode_steps': 300,
                'cube_fixed': True,
                'reward_multiplier': 1.2
            },
            3: {
                'name': 'CONTACT_LEARNING',
                'description': 'Apprendre à toucher le cube',
                'max_phases': 3,  # STABILIZE + APPROACH + CONTACT
                'success_threshold': 40.0,
                'episodes_required': 4,
                'max_episode_steps': 400,
                'cube_fixed': False,  # Cube peut bouger
                'reward_multiplier': 1.5
            },
            4: {
                'name': 'FULL_GRASPING',
                'description': 'Grasping complet',
                'max_phases': 6,  # Toutes les phases
                'success_threshold': 60.0,
                'episodes_required': 3,
                'max_episode_steps': 500,
                'cube_fixed': False,
                'reward_multiplier': 2.0
            },
            5: {
                'name': 'MASTER_LEVEL',
                'description': 'Grasping avec perturbations',
                'max_phases': 6,
                'success_threshold': 80.0,
                'episodes_required': 3,
                'max_episode_steps': 500,
                'cube_fixed': False,
                'reward_multiplier': 2.5,
                'add_noise': True,  # Ajouter du bruit aux observations
                'cube_variations': True  # Positions de cube variables
            }
        }
        
        # État du curriculum
        self.current_level = 1
        self.consecutive_successes = 0
        self.level_episodes = 0
        self.level_start_time = time.time()
        self.performance_history = []
        
        # Phases de grasping
        self.PHASES = {
            'STABILIZE': 0,
            'APPROACH': 1, 
            'CONTACT': 2,
            'GRASP': 3,
            'LIFT': 4,
            'HOLD': 5
        }
        
        # Configuration des phases selon le niveau de curriculum
        self._update_phase_config()
        
        # Initialisation du modèle
        self._setup_model()
        self._identify_components()
        self._setup_spaces()
        
        # Métriques et état
        self.episode_step = 0
        self.cube_initial_pos = np.array([0.3, 0.0, 0.05])
        self.stability_count = 0
        self.contact_count = 0
        self.successful_grasp = False
        self.cube_lifted = False
        
        # Historique pour détection de stabilité
        self.velocity_history = []
        self.position_history = []
        self.max_history = 10
        
        print(f"🎓 CurriculumGraspEnv initialisé!")
        print(f"📚 Niveau de curriculum: {self.current_level} - {self.curriculum_levels[self.current_level]['name']}")
        print(f"📖 Description: {self.curriculum_levels[self.current_level]['description']}")
        
    def _update_phase_config(self):
        """Met à jour la configuration des phases selon le niveau de curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        
        # Durées des phases adaptées au niveau
        if self.current_level == 1:  # Stabilisation uniquement
            self.phase_durations = {
                'STABILIZE': 200,
                'APPROACH': 0, 'CONTACT': 0, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0
            }
        elif self.current_level == 2:  # Stabilisation + Approche
            self.phase_durations = {
                'STABILIZE': 80,
                'APPROACH': 220,
                'CONTACT': 0, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0
            }
        elif self.current_level == 3:  # + Contact
            self.phase_durations = {
                'STABILIZE': 60,
                'APPROACH': 180,
                'CONTACT': 160,
                'GRASP': 0, 'LIFT': 0, 'HOLD': 0
            }
        else:  # Grasping complet
            self.phase_durations = {
                'STABILIZE': 80,
                'APPROACH': 120,
                'CONTACT': 40,
                'GRASP': 60,
                'LIFT': 50,
                'HOLD': 80
            }
        
        self.max_episode_steps = level_config['max_episode_steps']
        
    def _setup_model(self):
        """Configuration du modèle avec physique ultra-optimisée"""
        try:
            # Changer vers le répertoire du modèle pour les chemins relatifs
            original_cwd = os.getcwd()
            model_dir = os.path.dirname(self.model_path_str)
            os.chdir(model_dir)
            
            try:
                # Lire le fichier XML original
                with open(os.path.basename(self.model_path_str), 'r') as f:
                    xml_content = f.read()
                
                # Appliquer corrections physiques ultra-stables
                xml_content = self._apply_ultra_physics_fixes(xml_content)
                
                # Créer fichier temporaire dans le même répertoire
                self.temp_model_path = os.path.join(model_dir, 'temp_curriculum_model.xml')
                with open(self.temp_model_path, 'w') as f:
                    f.write(xml_content)
                
                # Charger le modèle
                self.model = mujoco.MjModel.from_xml_path('temp_curriculum_model.xml')
                self.data = mujoco.MjData(self.model)
                
                print("✅ Modèle chargé avec physique ultra-stable")
                print(f"  - DOFs: {self.model.nv}")
                print(f"  - Actuateurs: {self.model.nu}")
                print(f"  - Capteurs: {self.model.nsensor}")
                print(f"  - Timestep: {self.model.opt.timestep}")
                
            finally:
                # Revenir au répertoire original
                os.chdir(original_cwd)
            
        except Exception as e:
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _apply_ultra_physics_fixes(self, xml_content: str) -> str:
        """Applique des corrections physiques ultra-stables"""
        
        # 1. Timestep ultra-petit pour stabilité maximale
        xml_content = xml_content.replace(
            'timestep="0.002"',
            'timestep="0.0005"'
        )
        
        # 2. Augmenter les itérations pour convergence
        xml_content = xml_content.replace(
            'iterations="200"',
            'iterations="500"'
        )
        
        # 3. Tolérance ultra-stricte
        xml_content = xml_content.replace(
            'tolerance="1e-8"',
            'tolerance="1e-12"'
        )
        
        # 4. Augmenter le damping des bras pour stabilité
        arm_joints = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        for joint in arm_joints:
            # Augmenter kv (damping) pour les bras
            xml_content = xml_content.replace(
                f'name="act_{joint}" joint="{joint}" gear="1" kp="100" kv="10"',
                f'name="act_{joint}" joint="{joint}" gear="1" kp="120" kv="25"'
            )
        
        # 5. Configurer les groupes de collision pour assurer les collisions physiques
        finger_geom_names = ['left_index', 'left_middle', 'left_ring', 'left_thumb', 
                            'right_index', 'right_middle', 'right_ring', 'right_thumb']
        
        for finger in finger_geom_names:
            # Assurer que les doigts ont des collisions avec table et cube
            if f'mesh="{finger}' in xml_content:
                # Ajouter contype/conaffinity si pas présent
                xml_content = xml_content.replace(
                    f'<geom type="mesh" mesh="{finger}',
                    f'<geom type="mesh" mesh="{finger}" contype="4" conaffinity="7"'
                )
        
        # 6. Améliorer la friction de la table et du cube
        xml_content = xml_content.replace(
            'friction="1.0 0.1 0.05"',
            'friction="2.0 0.3 0.1"'  # Friction plus élevée
        )
        
        xml_content = xml_content.replace(
            'friction="1.5 0.2 0.1"',
            'friction="2.5 0.4 0.2"'  # Friction cube plus élevée
        )
        
        return xml_content
    
    def _identify_components(self):
        """Identifie les composants du robot"""
        # Joints des bras
        self.arm_joint_ids = []
        self.finger_joint_ids = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                if any(x in joint_name for x in ['shoulder', 'elbow', 'wrist']):
                    self.arm_joint_ids.append(i)
                elif any(x in joint_name for x in ['finger', 'thumb', 'index', 'middle', 'ring']):
                    self.finger_joint_ids.append(i)
        
        # Identifiants du cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Sites des doigts pour détection de contact
        self.finger_sites = []
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name and 'tip' in site_name:
                self.finger_sites.append(i)
        
        print(f"✅ Composants identifiés:")
        print(f"  - Joints bras: {len(self.arm_joint_ids)}")
        print(f"  - Joints doigts: {len(self.finger_joint_ids)}")
        print(f"  - Sites doigts: {len(self.finger_sites)}")
    
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation"""
        # Espace d'action: bras (14) + doigts (8) = 22 actions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32
        )
        
        # Espace d'observation: positions (37) + vitesses (37) + cube (7) + phase (1) + métriques (6) = 88
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32
        )
    
    def update_curriculum_level(self, episode_reward: float, episode_success: bool):
        """Met à jour le niveau de curriculum selon les performances"""
        current_config = self.curriculum_levels[self.current_level]
        
        # Enregistrer les performances
        self.performance_history.append({
            'episode': self.level_episodes,
            'reward': episode_reward,
            'success': episode_success,
            'level': self.current_level
        })
        
        # Garder seulement les 50 dernières performances
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
        
        self.level_episodes += 1
        
        # Vérifier si l'épisode est un succès
        if episode_reward >= current_config['success_threshold']:
            self.consecutive_successes += 1
            print(f"✅ Succès! ({self.consecutive_successes}/{current_config['episodes_required']}) - Récompense: {episode_reward:.2f}")
        else:
            self.consecutive_successes = 0
            print(f"📈 Progression - Récompense: {episode_reward:.2f} (Objectif: {current_config['success_threshold']:.2f})")
        
        # Vérifier si on peut passer au niveau suivant
        if (self.consecutive_successes >= current_config['episodes_required'] and 
            self.current_level < len(self.curriculum_levels)):
            
            self.current_level += 1
            self.consecutive_successes = 0
            self.level_episodes = 0
            self.level_start_time = time.time()
            
            # Mettre à jour la configuration des phases
            self._update_phase_config()
            
            new_config = self.curriculum_levels[self.current_level]
            print(f"\n🎉 NIVEAU SUPÉRIEUR ATTEINT!")
            print(f"📚 Niveau {self.current_level}: {new_config['name']}")
            print(f"📖 Description: {new_config['description']}")
            print(f"🎯 Objectif: {new_config['success_threshold']:.1f} points")
            print(f"📊 Phases actives: {new_config['max_phases']}")
            
        # Statistiques du niveau actuel
        if self.level_episodes % 10 == 0:
            recent_rewards = [p['reward'] for p in self.performance_history[-10:] if p['level'] == self.current_level]
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                print(f"📊 Niveau {self.current_level} - Épisode {self.level_episodes}")
                print(f"   Récompense moyenne (10 derniers): {avg_reward:.2f}")
                print(f"   Succès consécutifs: {self.consecutive_successes}")
    
    def reset(self, seed=None, options=None):
        """Remet l'environnement à zéro selon le niveau de curriculum"""
        if seed is not None:
            np.random.seed(seed)
        
        # Réinitialiser l'état
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale du cube selon le niveau de curriculum
        level_config = self.curriculum_levels[self.current_level]
        
        if level_config.get('cube_variations', False):
            # Positions variables pour niveau avancé
            cube_offset = np.random.uniform(-0.05, 0.05, 3)
            cube_offset[2] = abs(cube_offset[2])  # Garder Z positif
        else:
            # Position fixe ou légèrement variable
            cube_offset = np.random.uniform(-0.01, 0.01, 3)
            cube_offset[2] = abs(cube_offset[2]) * 0.5
        
        self.cube_initial_pos = np.array([0.3, 0.0, 0.05]) + cube_offset
        
        if self.cube_body_id >= 0:
            self.data.qpos[self.model.nq - 7:self.model.nq - 4] = self.cube_initial_pos
        
        # Réinitialiser les variables d'état
        self.current_phase = self.PHASES['STABILIZE']
        self.phase_timer = 0
        self.episode_step = 0
        self.stability_count = 0
        self.contact_count = 0
        self.successful_grasp = False
        self.cube_lifted = False
        
        # Réinitialiser historiques
        self.velocity_history = []
        self.position_history = []
        
        # Position initiale des bras (position neutre avec légère variation)
        arm_initial_pos = np.array([0.0, 0.1, 0.0, -0.5, 0.0, 0.3, 0.0] * 2)  # 2 bras
        if level_config.get('add_noise', False):
            # Ajouter du bruit pour niveau avancé
            arm_noise = np.random.uniform(-0.1, 0.1, len(arm_initial_pos))
            arm_initial_pos += arm_noise
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            if i < len(arm_initial_pos):
                self.data.qpos[joint_id] = arm_initial_pos[i]
        
        # Doigts ouverts initialement
        for joint_id in self.finger_joint_ids:
            self.data.qpos[joint_id] = 0.1
        
        # Simuler quelques pas pour stabiliser
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Exécute une action dans l'environnement avec curriculum adaptatif"""
        self.episode_step += 1
        self.phase_timer += 1
        
        # Limiter les actions pour éviter les mouvements brusques
        action = np.clip(action, -1.0, 1.0)
        
        # Appliquer scaling adaptatif selon la phase et le niveau de curriculum
        action = self._apply_curriculum_scaling(action)
        
        # Appliquer les actions avec smooth control
        self._apply_smooth_actions(action)
        
        # Simulation physique
        mujoco.mj_step(self.model, self.data)
        
        # Vérifier et corriger les instabilités
        self._check_stability()
        
        # Mettre à jour la phase selon le curriculum
        self._update_phase_curriculum()
        
        # Calculer observation et récompense
        observation = self._get_observation()
        reward = self._calculate_curriculum_reward()
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_curriculum_scaling(self, action):
        """Applique un scaling adaptatif selon la phase et le niveau de curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        phase_name = self._get_phase_name()
        
        # Scaling de base selon la phase
        if phase_name == 'STABILIZE':
            base_scale = 0.05
        elif phase_name == 'APPROACH':
            base_scale = 0.15
        elif phase_name == 'CONTACT':
            base_scale = 0.08
        elif phase_name == 'GRASP':
            # Focus sur fermeture des doigts
            arm_scale = 0.02
            finger_scale = 0.3
            scaled_action = action.copy()
            scaled_action[:14] *= arm_scale
            scaled_action[14:] *= finger_scale
            return scaled_action
        elif phase_name == 'LIFT':
            base_scale = 0.12
        else:  # HOLD
            base_scale = 0.03
        
        # Ajustement selon le niveau de curriculum
        if self.current_level == 1:
            # Niveau débutant: mouvements très lents
            curriculum_multiplier = 0.5
        elif self.current_level == 2:
            # Niveau intermédiaire: mouvements modérés
            curriculum_multiplier = 0.8
        else:
            # Niveau avancé: mouvements normaux
            curriculum_multiplier = 1.0
        
        final_scale = base_scale * curriculum_multiplier
        
        # Ajouter du bruit pour niveau avancé
        if level_config.get('add_noise', False):
            noise = np.random.normal(0, 0.01, action.shape)
            action = action + noise
        
        return action * final_scale
    
    def _apply_smooth_actions(self, action):
        """Applique les actions avec contrôle de vitesse"""
        # Actions pour les bras (0-13)
        arm_actions = action[:14]
        # Actions pour les doigts (14-21)
        finger_actions = action[14:22]
        
        # Appliquer aux bras avec limitation de vitesse
        for i, joint_id in enumerate(self.arm_joint_ids):
            if i < len(arm_actions):
                current_pos = self.data.qpos[joint_id]
                target_pos = current_pos + arm_actions[i] * 0.1  # Petit incrément
                
                # Limiter la vitesse de changement selon le niveau de curriculum
                max_change = 0.03 if self.current_level <= 2 else 0.05
                if abs(target_pos - current_pos) > max_change:
                    target_pos = current_pos + np.sign(target_pos - current_pos) * max_change
                
                self.data.ctrl[i] = target_pos
        
        # Appliquer aux doigts
        finger_offset = len(self.arm_joint_ids)
        for i, joint_id in enumerate(self.finger_joint_ids):
            if i < len(finger_actions):
                current_pos = self.data.qpos[joint_id]
                target_pos = current_pos + finger_actions[i] * 0.2
                
                # Limiter dans les bounds du joint
                target_pos = np.clip(target_pos, 0.0, 1.2)
                
                self.data.ctrl[finger_offset + i] = target_pos
    
    def _check_stability(self):
        """Vérifie et corrige les instabilités"""
        # Vérifier les NaN/Inf
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)):
            print("⚠️ Instabilité détectée - récupération...")
            mujoco.mj_resetData(self.model, self.data)
            return
        
        # Vérifier les vitesses excessives
        max_velocity = np.max(np.abs(self.data.qvel))
        if max_velocity > 10.0:
            # Réduire toutes les vitesses
            self.data.qvel *= 0.5
            if self.episode_step % 50 == 0:  # Afficher moins souvent
                print(f"⚠️ Vitesse excessive ({max_velocity:.2f}) - réduction appliquée")
        
        # Historique pour détection de stabilité
        arm_velocities = [self.data.qvel[i] for i in self.arm_joint_ids]
        mean_arm_velocity = np.mean(np.abs(arm_velocities))
        
        self.velocity_history.append(mean_arm_velocity)
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
        
        # Compter les steps stables
        if mean_arm_velocity < 0.1:
            self.stability_count += 1
        else:
            self.stability_count = 0
    
    def _update_phase_curriculum(self):
        """Met à jour la phase selon la progression et le curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        max_phases = level_config['max_phases']
        
        # Limiter les phases selon le niveau de curriculum
        if self.current_phase >= max_phases:
            return  # Ne pas dépasser le niveau autorisé
        
        phase_name = self._get_phase_name()
        should_advance = False
        
        if phase_name == 'STABILIZE':
            # Critères de stabilisation adaptés au niveau
            stability_threshold = 15 if self.current_level <= 2 else 20
            if self.stability_count > stability_threshold or self.phase_timer >= self.phase_durations['STABILIZE']:
                should_advance = True
                
        elif phase_name == 'APPROACH':
            # Avancer si proche du cube ou timer écoulé
            cube_pos = self._get_cube_position()
            hand_pos = self._get_hand_center()
            distance = np.linalg.norm(cube_pos - hand_pos)
            distance_threshold = 0.2 if self.current_level <= 2 else 0.15
            if distance < distance_threshold or self.phase_timer >= self.phase_durations['APPROACH']:
                should_advance = True
                
        elif phase_name == 'CONTACT':
            # Avancer si contact détecté ou timer écoulé
            if self._detect_finger_contact() or self.phase_timer >= self.phase_durations['CONTACT']:
                should_advance = True
                
        elif phase_name == 'GRASP':
            # Avancer si prise stable ou timer écoulé
            if self._check_grasp_stability() or self.phase_timer >= self.phase_durations['GRASP']:
                should_advance = True
                self.successful_grasp = True
                
        elif phase_name == 'LIFT':
            # Avancer si cube soulevé ou timer écoulé
            if self._is_cube_lifted() or self.phase_timer >= self.phase_durations['LIFT']:
                should_advance = True
                self.cube_lifted = True
        
        # Avancer à la phase suivante si autorisé par le curriculum
        if should_advance and self.current_phase < min(5, max_phases - 1):
            self.current_phase += 1
            self.phase_timer = 0
            phase_name = self._get_phase_name()
            if self.episode_step % 100 == 0:  # Afficher moins souvent
                print(f"📈 Transition vers phase: {phase_name}")
    
    def _calculate_curriculum_reward(self):
        """Calcule une récompense adaptée au niveau de curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        reward_multiplier = level_config['reward_multiplier']
        
        reward = 0.0
        phase_name = self._get_phase_name()
        
        # Récompense de base pour chaque step (éviter terminaison immédiate)
        reward += 0.1
        
        # Bonus de stabilité (très important)
        if self.stability_count > 0:
            reward += 0.2 * reward_multiplier
        if self.stability_count > 10:
            reward += 0.3 * reward_multiplier
        
        # Récompenses spécifiques selon le niveau de curriculum
        if self.current_level == 1:  # Stabilisation uniquement
            if phase_name == 'STABILIZE':
                arm_velocities = [abs(self.data.qvel[i]) for i in self.arm_joint_ids]
                stability_reward = max(0, 1.0 - np.mean(arm_velocities))
                reward += stability_reward * 5.0  # Récompense élevée pour stabilité
                
                # Bonus pour maintenir la stabilité longtemps
                if self.stability_count > 30:
                    reward += 3.0
                if self.stability_count > 50:
                    reward += 5.0
        
        elif self.current_level == 2:  # + Approche
            if phase_name == 'STABILIZE':
                arm_velocities = [abs(self.data.qvel[i]) for i in self.arm_joint_ids]
                stability_reward = max(0, 1.0 - np.mean(arm_velocities))
                reward += stability_reward * 3.0
                
            elif phase_name == 'APPROACH':
                # Récompenser l'approche du cube
                cube_pos = self._get_cube_position()
                hand_pos = self._get_hand_center()
                distance = np.linalg.norm(cube_pos - hand_pos)
                approach_reward = max(0, 1.0 - distance / 0.5)
                reward += approach_reward * 4.0
        
        elif self.current_level == 3:  # + Contact
            # Récompenses pour toutes les phases activées
            if phase_name == 'STABILIZE':
                arm_velocities = [abs(self.data.qvel[i]) for i in self.arm_joint_ids]
                stability_reward = max(0, 1.0 - np.mean(arm_velocities))
                reward += stability_reward * 2.0
                
            elif phase_name == 'APPROACH':
                cube_pos = self._get_cube_position()
                hand_pos = self._get_hand_center()
                distance = np.linalg.norm(cube_pos - hand_pos)
                approach_reward = max(0, 1.0 - distance / 0.5)
                reward += approach_reward * 3.0
                
            elif phase_name == 'CONTACT':
                if self._detect_finger_contact():
                    reward += 8.0 * reward_multiplier
                    self.contact_count += 1
        
        else:  # Grasping complet (niveau 4+)
            # Système de récompenses complet
            if phase_name == 'STABILIZE':
                arm_velocities = [abs(self.data.qvel[i]) for i in self.arm_joint_ids]
                stability_reward = max(0, 1.0 - np.mean(arm_velocities))
                reward += stability_reward * 2.0
                
            elif phase_name == 'APPROACH':
                cube_pos = self._get_cube_position()
                hand_pos = self._get_hand_center()
                distance = np.linalg.norm(cube_pos - hand_pos)
                approach_reward = max(0, 1.0 - distance / 0.5)
                reward += approach_reward * 3.0
                
            elif phase_name == 'CONTACT':
                if self._detect_finger_contact():
                    reward += 5.0 * reward_multiplier
                    self.contact_count += 1
                
            elif phase_name == 'GRASP':
                if self._check_grasp_stability():
                    reward += 8.0 * reward_multiplier
                
            elif phase_name == 'LIFT':
                cube_height = self._get_cube_position()[2]
                lift_reward = max(0, (cube_height - 0.05) / 0.1)
                reward += lift_reward * 10.0 * reward_multiplier
                
            elif phase_name == 'HOLD':
                if self._is_cube_lifted() and self._check_grasp_stability():
                    reward += 15.0 * reward_multiplier
        
        # Bonus de progression de phase
        reward += self.current_phase * 2.0 * reward_multiplier
        
        # Pénalités légères pour éviter comportements indésirables
        if np.max(np.abs(self.data.qvel)) > 5.0:
            reward -= 0.5  # Pénalité pour mouvements brusques
        
        # Assurer que la récompense reste dans un range raisonnable
        reward = np.clip(reward, -10.0, 100.0)
        
        return reward
    
    def _get_observation(self):
        """Construit l'observation de l'état avec info curriculum"""
        obs = []
        
        # Positions des joints (37)
        obs.extend(self.data.qpos.copy())
        
        # Vitesses des joints (37)  
        obs.extend(self.data.qvel.copy())
        
        # Position et orientation du cube (7)
        if self.cube_body_id >= 0:
            cube_pos = self.data.xpos[self.cube_body_id].copy()
            cube_quat = self.data.xquat[self.cube_body_id].copy()
            obs.extend(cube_pos)
            obs.extend(cube_quat)
        else:
            obs.extend([0.0] * 7)
        
        # Phase actuelle (1)
        obs.append(float(self.current_phase))
        
        # Métriques additionnelles (6)
        obs.append(float(self.stability_count))
        obs.append(float(self.contact_count))
        obs.append(float(self.successful_grasp))
        obs.append(float(self.cube_lifted))
        obs.append(float(self.phase_timer))
        obs.append(float(self.current_level))  # Niveau de curriculum
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        """Retourne les informations de debug avec curriculum"""
        return {
            'phase': self._get_phase_name(),
            'phase_timer': self.phase_timer,
            'stability_count': self.stability_count,
            'contact_count': self.contact_count,
            'successful_grasp': self.successful_grasp,
            'cube_lifted': self.cube_lifted,
            'cube_position': self._get_cube_position().tolist(),
            'episode_step': self.episode_step,
            'curriculum_level': self.current_level,
            'curriculum_name': self.curriculum_levels[self.current_level]['name'],
            'consecutive_successes': self.consecutive_successes,
            'level_episodes': self.level_episodes
        }
    
    def _check_termination(self):
        """Vérifie les conditions de terminaison selon le curriculum"""
        level_config = self.curriculum_levels[self.current_level]
        max_phases = level_config['max_phases']
        
        # Succès selon le niveau de curriculum
        if self.current_level == 1:  # Stabilisation
            if self.stability_count > 50 and self.phase_timer > 100:
                return True
        elif self.current_level == 2:  # + Approche
            cube_pos = self._get_cube_position()
            hand_pos = self._get_hand_center()
            distance = np.linalg.norm(cube_pos - hand_pos)
            if distance < 0.15 and self.current_phase >= 1:
                return True
        elif self.current_level == 3:  # + Contact
            if self.contact_count > 5 and self.current_phase >= 2:
                return True
        else:  # Grasping complet
            if (self.current_phase == self.PHASES['HOLD'] and 
                self.cube_lifted and 
                self.phase_timer > 30):
                return True
        
        # Échec: cube tombé de la table
        cube_pos = self._get_cube_position()
        if cube_pos[2] < 0.0 or abs(cube_pos[0]) > 1.0 or abs(cube_pos[1]) > 1.0:
            return True
            
        return False
    
    # Méthodes utilitaires (identiques à la version précédente)
    def _get_phase_name(self):
        phase_names = ['STABILIZE', 'APPROACH', 'CONTACT', 'GRASP', 'LIFT', 'HOLD']
        return phase_names[self.current_phase]
    
    def _get_cube_position(self):
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id].copy()
        return self.cube_initial_pos
    
    def _get_hand_center(self):
        if len(self.finger_sites) > 0:
            positions = [self.data.site_xpos[site_id] for site_id in self.finger_sites]
            return np.mean(positions, axis=0)
        return np.array([0.0, 0.0, 0.0])
    
    def _detect_finger_contact(self):
        cube_pos = self._get_cube_position()
        for site_id in self.finger_sites:
            finger_pos = self.data.site_xpos[site_id]
            distance = np.linalg.norm(cube_pos - finger_pos)
            if distance < 0.05:
                return True
        return False
    
    def _check_grasp_stability(self):
        if not self._detect_finger_contact():
            return False
        cube_vel = self.data.cvel[self.cube_body_id] if self.cube_body_id >= 0 else np.zeros(6)
        return np.linalg.norm(cube_vel) < 0.1
    
    def _is_cube_lifted(self):
        cube_pos = self._get_cube_position()
        return cube_pos[2] > 0.08
    
    def render(self):
        if self.render_mode == "human":
            pass
        return None
    
    def close(self):
        """Nettoie les ressources"""
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
            'performance_history': self.performance_history[-10:]  # 10 dernières performances
        }
