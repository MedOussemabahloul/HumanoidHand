#!/usr/bin/env python3
"""
🤖 ENVIRONNEMENT GRASPING OPTIMISÉ - INSPIRÉ DU COLLÈGUE
========================================================

Environnement qui s'inspire des bonnes pratiques du collègue tout en gardant
notre propre approche professionnelle:

✅ INSPIRATIONS DU COLLÈGUE:
- Scaling adaptatif des actions selon la distance
- Reset des contrôles à chaque step  
- Position fixe du cube pour stabilité
- Assistance contextuelle au grasping

✅ NOTRE APPROCHE AMÉLIORÉE:
- Curriculum learning progressif
- Gestion robuste des erreurs NaN/inf
- Récompenses équilibrées et motivantes
- Mouvements fluides et professionnels
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import tempfile
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

class OptimizedGraspEnv(gym.Env):
    """
    🤖 Environnement optimisé pour le grasping robotique
    
    INSPIRATIONS DU COLLÈGUE:
    - Scaling adaptatif: ARM_SCALE = 0.4 si dist > 0.08 else 0.2
    - Reset contrôles: self.data.ctrl[:] = 0.0 
    - Position cube fixe: [0.18, 0.0, 0.04]
    - Assistance: aide quand 2+ doigts touchent
    
    NOTRE VALEUR AJOUTÉE:
    - Curriculum learning avec phases progressives
    - Gestion robuste des NaN/inf
    - Récompenses motivantes et équilibrées
    - Mouvements fluides et naturels
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 render_mode: str = "rgb_array",
                 max_episode_steps: int = 500,
                 curriculum_level: int = 1,
                 enable_smooth_movements: bool = True):
        
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_level = curriculum_level
        self.enable_smooth_movements = enable_smooth_movements
        
        # Logger
        self._setup_logging()
        
        # Configuration headless comme le collègue
        self._setup_headless_rendering()
        
        # Modèle MuJoCo optimisé - validation du chemin
        self.model_path = model_path or self._validate_xml_path()
        self._load_mujoco_model()
        
        # Configuration des composants
        self._setup_robot_components()
        self._setup_spaces()
        
        # Variables d'état
        self._reset_episode_vars()
        
        # Historique pour mouvements fluides
        self.action_history = []
        self.max_action_history = 5
        
        self.logger.info(f"🤖 Environnement optimisé initialisé (niveau curriculum: {curriculum_level})")
    
    def _setup_logging(self):
        """Configure le logging"""
        self.logger = logging.getLogger("OptimizedGrasp")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_headless_rendering(self):
        """Configure le rendu headless comme le collègue"""
        import os
        
        # Configuration EGL comme le collègue
        os.environ["MUJOCO_GL"] = "egl"
        
        # Autres configurations pour éviter les erreurs graphiques
        os.environ.setdefault("EGL_DEVICE_ID", "0")
        
        self.logger.info("🖥️ Configuration headless activée (EGL)")
    
    def _validate_xml_path(self) -> str:
        """Valide et retourne le chemin XML approprié"""
        
        # Essayer d'abord le modèle optimisé
        optimized_path = "/workspace/results/g1_combined_optimized.xml"
        if Path(optimized_path).exists():
            self.logger.info(f"✅ Utilisation du modèle optimisé: {optimized_path}")
            return optimized_path
        
        # Fallback sur le modèle du collègue
        colleague_path = "/workspace/results/g1_combined_ultra_stable.xml"
        if Path(colleague_path).exists():
            self.logger.info(f"✅ Utilisation du modèle du collègue: {colleague_path}")
            return colleague_path
        
        # Dernier recours: créer un modèle minimal
        self.logger.warning("⚠️ Création d'un modèle minimal temporaire")
        return self._create_minimal_model()
    
    def _create_minimal_model(self) -> str:
        """Crée un modèle minimal sans conflits d'inclusion"""
        
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="minimal_grasp">
    <compiler angle="radian"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" 
            solver="PGS" iterations="50" tolerance="1e-10"/>
    
    <default>
        <geom friction="0.8 0.1 0.05"/>
        <joint damping="1.0"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <worldbody>
        <light name="light" pos="0 0 2"/>
        
        <body name="table" pos="0 0 0.4">
            <geom type="box" size="0.5 0.5 0.05" rgba="0.8 0.6 0.4 1"/>
        </body>
        
        <body name="cube" pos="0.18 0 0.04">
            <joint name="cube:joint" type="free"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" 
                  rgba="0.2 0.8 0.2 1" friction="1.0"/>
            <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
        </body>
        
        <body name="robot_base" pos="0 0 0.5">
            <body name="arm" pos="0 -0.1 0.1">
                <joint name="shoulder" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                <geom type="capsule" size="0.03 0.1" rgba="0.7 0.7 0.7 1"/>
                
                <body name="hand" pos="0 0 -0.15">
                    <joint name="wrist" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="box" size="0.02 0.03 0.01" rgba="0.9 0.7 0.5 1"/>
                    
                    <body name="finger1" pos="0.02 0.02 0">
                        <joint name="finger1_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                        <geom name="finger1_geom" type="capsule" size="0.008 0.02" 
                              rgba="0.9 0.7 0.5 1" friction="1.5"/>
                    </body>
                    
                    <body name="finger2" pos="0.02 -0.02 0">
                        <joint name="finger2_joint" type="hinge" axis="0 1 0" range="0 1.57"/>
                        <geom name="finger2_geom" type="capsule" size="0.008 0.02" 
                              rgba="0.9 0.7 0.5 1" friction="1.5"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="shoulder_motor" joint="shoulder" kp="10" kv="3"/>
        <position name="wrist_motor" joint="wrist" kp="8" kv="2"/>
        <position name="finger1_motor" joint="finger1_joint" kp="5" kv="1"/>
        <position name="finger2_motor" joint="finger2_joint" kp="5" kv="1"/>
    </actuator>
</mujoco>'''
        
        # Sauvegarder le modèle
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(model_xml)
            return f.name
    
    def _load_mujoco_model(self):
        """Charge le modèle MuJoCo avec gestion d'erreurs"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Configuration du rendu (avec gestion d'erreurs)
            self.renderer = None
            if self.render_mode == "rgb_array":
                try:
                    self.renderer = mujoco.Renderer(self.model, width=640, height=480)
                    self.logger.info("✅ Renderer initialisé")
                except Exception as e:
                    self.logger.warning(f"⚠️ Impossible d'initialiser le renderer: {e}")
                    self.render_mode = "none"
            
            self.logger.info(f"✅ Modèle MuJoCo chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _setup_robot_components(self):
        """Configure les composants du robot"""
        
        # Identifier les actuateurs (inspiré du collègue mais plus robuste)
        self.arm_actuators = []
        self.finger_actuators = []
        
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                if any(joint in name for joint in ["shoulder", "elbow", "wrist"]):
                    self.arm_actuators.append(i)
                elif any(finger in name for finger in ["thumb", "index", "middle"]):
                    self.finger_actuators.append(i)
        
        self.all_actuators = self.arm_actuators + self.finger_actuators
        
        self.logger.info(f"✅ Composants configurés: {len(self.arm_actuators)} bras, {len(self.finger_actuators)} doigts")
    
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation"""
        
        # Espace d'action pour tous les actuateurs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.all_actuators),),
            dtype=np.float32
        )
        
        # Espace d'observation robuste
        obs_dim = self.model.nq + self.model.nv + 12  # qpos + qvel + infos cube/main
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0,  # Limites raisonnables pour éviter inf
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.logger.info(f"✅ Espaces configurés: Action ({self.action_space.shape[0]},), Obs ({obs_dim},)")
    
    def _reset_episode_vars(self):
        """Reset des variables d'épisode"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.best_distance = float('inf')
        self.contact_history = []
        self.action_history = []
        
        # Métriques de curriculum
        self.success_contacts = 0
        self.stable_grasp_duration = 0
    
    def reset(self, seed=None, options=None):
        """Reset de l'environnement avec position cube fixe comme le collègue"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Position cube FIXE comme le collègue: [0.18, 0.0, 0.04]
        try:
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube:joint")
            if cube_joint_id >= 0:
                cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
                
                # Position fixe
                fixed_cube_pos = np.array([0.18, 0.0, 0.04])
                start = cube_qpos_addr
                end = min(cube_qpos_addr + 3, len(self.data.qpos))
                self.data.qpos[start:end] = fixed_cube_pos[:end-start]
                
                # Orientation fixe
                fixed_cube_quat = np.array([1, 0, 0, 0])
                start = cube_qpos_addr + 3
                end = min(cube_qpos_addr + 7, len(self.data.qpos))
                if end > start:
                    self.data.qpos[start:end] = fixed_cube_quat[:end-start]
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de fixer position cube: {e}")
        
        # Reset variables
        self._reset_episode_vars()
        
        # Observation initiale
        obs = self._get_obs()
        
        return obs, {}
    
    def step(self, action):
        """Step inspiré du collègue avec nos améliorations"""
        
        # Validation et nettoyage de l'action
        action = self._sanitize_action(action)
        
        # Séparation bras/doigts comme le collègue
        n_arm = len(self.arm_actuators)
        arm_action = action[:n_arm] if n_arm > 0 else np.array([])
        finger_action = action[n_arm:] if len(action) > n_arm else np.array([])
        
        # Calcul des positions et distances
        positions = self._get_positions()
        dist = positions['palm_to_cube_dist']
        
        # SCALING ADAPTATIF comme le collègue mais plus fluide
        arm_scale = self._get_adaptive_arm_scale(dist)
        finger_scale = self._get_adaptive_finger_scale(dist, positions)
        
        # Lissage des mouvements (notre valeur ajoutée)
        if self.enable_smooth_movements:
            action = self._apply_movement_smoothing(action)
        
        # RESET CONTRÔLES comme le collègue (clé du succès!)
        self.data.ctrl[:] = 0.0
        
        # Application des actions avec scaling
        if len(self.arm_actuators) > 0 and len(arm_action) > 0:
            self.data.ctrl[self.arm_actuators] = arm_action * arm_scale
        
        if len(self.finger_actuators) > 0 and len(finger_action) > 0:
            self.data.ctrl[self.finger_actuators] = finger_action * finger_scale
        
        # ASSISTANCE AU GRASPING comme le collègue
        self._apply_grasp_assistance(positions)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Calcul récompense et observation
        obs = self._get_obs()
        reward = self._compute_reward(positions)
        terminated = self._check_termination(positions)
        
        # Mise à jour état
        self.current_step += 1
        self.episode_reward += reward
        
        # Info pour debugging
        info = {
            'distance': dist,
            'contact_count': positions['contact_count'],
            'cube_velocity': positions['cube_velocity'],
            'episode_step': self.current_step,
            'curriculum_level': self.curriculum_level,
            'arm_scale': arm_scale,
            'finger_scale': finger_scale
        }
        
        return obs, reward, terminated, False, info
    
    def _sanitize_action(self, action):
        """Nettoie l'action pour éviter NaN/inf"""
        action = np.array(action, dtype=np.float32)
        
        # Remplacer NaN/inf par 0
        action = np.where(np.isfinite(action), action, 0.0)
        
        # Clipper dans les limites
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _get_adaptive_arm_scale(self, distance):
        """Scaling adaptatif du bras comme le collègue mais plus fluide"""
        
        # Inspiration du collègue: ARM_SCALE = 0.4 si dist > 0.08 else 0.2
        # Notre amélioration: transition plus fluide
        
        if distance > 0.12:
            return 0.5  # Mouvement rapide pour approche lointaine
        elif distance > 0.08:
            return 0.4  # Comme le collègue
        elif distance > 0.05:
            return 0.2  # Comme le collègue
        else:
            return 0.1  # Très fin pour positionnement précis
    
    def _get_adaptive_finger_scale(self, distance, positions):
        """Scaling adaptatif des doigts selon contexte"""
        
        base_scale = 0.7  # Comme le collègue
        
        # Ajustement selon curriculum
        curriculum_factor = min(1.0, self.curriculum_level * 0.2)
        
        # Réduction si très proche pour finesse
        if distance < 0.04:
            base_scale *= 0.6
        
        return base_scale * curriculum_factor
    
    def _apply_movement_smoothing(self, action):
        """Applique un lissage des mouvements pour fluidité"""
        
        # Ajouter à l'historique
        self.action_history.append(action.copy())
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)
        
        # Si on a assez d'historique, appliquer lissage
        if len(self.action_history) >= 3:
            # Moyenne pondérée avec plus de poids sur l'action courante
            weights = np.array([0.1, 0.3, 0.6])
            n_history = len(self.action_history)
            
            # Ajuster les poids selon la taille de l'historique
            if n_history < 3:
                weights = weights[:n_history]
            weights = weights / weights.sum()
            
            smoothed = np.zeros_like(action)
            for i, hist_action in enumerate(self.action_history):
                if i < len(weights):
                    smoothed += weights[i] * hist_action
            
            return smoothed
        
        return action
    
    def _get_positions(self):
        """Calcule toutes les positions nécessaires"""
        
        try:
            # Positions des objets
            cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            cube_pos = self.data.xpos[cube_id] if cube_id >= 0 else np.zeros(3)
            
            # Position de la main (adaptable selon le modèle)
            palm_pos = self._get_palm_position()
            
            # Positions des doigts (adaptable selon le modèle)
            finger_positions = self._get_finger_positions(palm_pos)
            
            # Distances
            palm_to_cube_dist = np.linalg.norm(palm_pos - cube_pos)
            
            # Vitesse du cube
            cube_velocity = np.linalg.norm(self.data.cvel[cube_id]) if cube_id >= 0 else 0.0
            
            # Contacts (inspiré du collègue)
            contact_count = self._count_finger_contacts()
            
            return {
                'cube_pos': cube_pos,
                'palm_pos': palm_pos,
                'finger_positions': finger_positions,
                'palm_to_cube_dist': palm_to_cube_dist,
                'cube_velocity': cube_velocity,
                'contact_count': contact_count
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur calcul positions: {e}")
            # Retour sécurisé
            return {
                'cube_pos': np.array([0.18, 0.0, 0.04]),
                'palm_pos': np.array([0.0, 0.0, 0.5]),
                'finger_positions': {},
                'palm_to_cube_dist': 0.5,
                'cube_velocity': 0.0,
                'contact_count': 0
            }
    
    def _get_palm_position(self):
        """Obtient la position de la main selon le modèle disponible"""
        
        # Essayer différents noms possibles pour la main
        palm_candidates = [
            "right_hand_index_1_link",  # Modèle complet
            "hand",                     # Modèle minimal
            "forearm",                  # Fallback
            "arm"                       # Dernier recours
        ]
        
        for name in palm_candidates:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    return self.data.xpos[body_id]
            except:
                continue
        
        # Position par défaut
        return np.array([0.0, 0.0, 0.5])
    
    def _get_finger_positions(self, palm_pos):
        """Obtient les positions des doigts selon le modèle disponible"""
        
        finger_positions = {}
        
        # Candidats pour les doigts selon le modèle
        finger_candidates = [
            # Modèle complet
            ["right_hand_thumb_2_link", "right_hand_index_2_link", "right_hand_middle_1_link"],
            # Modèle minimal
            ["finger1", "finger2"],
            # Fallback - utiliser la position de la main
            []
        ]
        
        for candidates in finger_candidates:
            found_fingers = {}
            for name in candidates:
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if body_id >= 0:
                        found_fingers[name] = self.data.xpos[body_id]
                except:
                    continue
            
            if found_fingers:
                return found_fingers
        
        # Fallback - pas de doigts détectés
        return {"default_finger": palm_pos}
    
    def _count_finger_contacts(self):
        """Compte les contacts des doigts avec le cube (adaptable selon modèle)"""
        
        contact_count = 0
        
        # Géométries possibles selon le modèle
        finger_geom_candidates = [
            # Modèle complet
            ["right_hand_thumb_2_geom", "right_hand_index_1_geom", "right_hand_middle_1_geom"],
            # Modèle minimal
            ["finger1_geom", "finger2_geom"],
            # Fallback - toute géométrie contenant "finger"
            []
        ]
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            try:
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if not name1 or not name2:
                    continue
                
                # Vérifier contact cube avec doigt
                cube_involved = name1 == "cube_geom" or name2 == "cube_geom"
                finger_involved = False
                
                # Essayer les différents patterns de doigts
                for candidates in finger_geom_candidates:
                    if candidates:
                        finger_involved = any(
                            name1 in candidates or name2 in candidates
                            for candidates in [candidates]
                        )
                        if finger_involved:
                            break
                
                # Fallback - chercher "finger" dans le nom
                if not finger_involved:
                    finger_involved = ("finger" in name1.lower() or 
                                     "finger" in name2.lower() or
                                     "thumb" in name1.lower() or 
                                     "thumb" in name2.lower() or
                                     "index" in name1.lower() or 
                                     "index" in name2.lower() or
                                     "middle" in name1.lower() or 
                                     "middle" in name2.lower())
                
                if cube_involved and finger_involved:
                    contact_count += 1
                    
            except:
                continue
        
        return contact_count
    
    def _apply_grasp_assistance(self, positions):
        """Assistance au grasping comme le collègue mais paramétrable"""
        
        dist = positions['palm_to_cube_dist']
        contact_count = positions['contact_count']
        
        # ASSISTANCE comme le collègue: si dist < 0.06 et 2+ contacts
        if dist < 0.06 and contact_count >= 2:
            # Assistance progressive selon curriculum
            assist_strength = 0.5 * min(1.0, self.curriculum_level * 0.3)
            
            # Appliquer assistance aux doigts
            if len(self.finger_actuators) > 0:
                self.data.ctrl[self.finger_actuators] += assist_strength
                self.data.ctrl[self.finger_actuators] = np.clip(
                    self.data.ctrl[self.finger_actuators], -1.0, 1.0
                )
            
            # Debug occasionnel
            if self.current_step % 50 == 0:
                self.logger.info(f"🤝 Assistance grasping activée (contacts: {contact_count})")
    
    def _compute_reward(self, positions):
        """Calcul de récompense inspiré du collègue mais équilibré"""
        
        dist = positions['palm_to_cube_dist']
        cube_vel = positions['cube_velocity']
        contact_count = positions['contact_count']
        
        # Base reward structure inspirée du collègue
        reward = 0.0
        
        # 1. Récompense de proximité (comme le collègue)
        reward += 5.0 / (1.0 + 20 * dist)
        
        # 2. Bonus de proximité (comme le collègue)
        if dist < 0.06:
            reward += 2.0
        
        # 3. Récompense de contact (inspirée du collègue mais améliorée)
        if contact_count == 0:
            grasp_quality = -0.5  # Légère pénalité
        elif contact_count == 1:
            grasp_quality = 0.2
        elif contact_count == 2:
            grasp_quality = 0.6
        else:  # 3+ contacts
            grasp_quality = 1.0 if cube_vel < 0.05 else 0.7
        
        reward += 8.0 * grasp_quality
        
        # 4. Pénalité vitesse (comme le collègue)
        reward -= 1.5 * min(1.0, cube_vel)
        
        # 5. Notre ajout: bonus curriculum
        curriculum_bonus = self.curriculum_level * 0.1
        reward += curriculum_bonus
        
        # 6. Pénalité temps modérée
        reward -= 0.003
        
        # 7. Bonus stabilité (notre ajout)
        if contact_count >= 2 and cube_vel < 0.02:
            self.stable_grasp_duration += 1
            if self.stable_grasp_duration > 10:
                reward += 0.5  # Bonus grasp stable
        else:
            self.stable_grasp_duration = 0
        
        # Mise à jour métriques
        if dist < self.best_distance:
            self.best_distance = dist
        
        # Debug occasionnel
        if self.current_step % 100 == 0:
            self.logger.info(
                f"[step {self.current_step}] dist: {dist:.3f}, "
                f"vel: {cube_vel:.3f}, contacts: {contact_count}, "
                f"grasp_quality: {grasp_quality:.2f}, reward: {reward:.2f}"
            )
        
        return float(reward)
    
    def _get_obs(self):
        """Observation robuste avec gestion NaN/inf"""
        
        try:
            # État de base
            base_state = np.concatenate([self.data.qpos, self.data.qvel])
            
            # Positions importantes
            positions = self._get_positions()
            cube_pos = positions['cube_pos']
            palm_pos = positions['palm_pos']
            relative_pos = cube_pos - palm_pos
            
            # Infos supplémentaires
            extra_info = np.array([
                positions['palm_to_cube_dist'],
                positions['cube_velocity'],
                float(positions['contact_count']),
                float(self.curriculum_level),
                float(self.current_step) / self.max_episode_steps,
                float(self.stable_grasp_duration)
            ])
            
            # Assemblage
            obs = np.concatenate([base_state, cube_pos, palm_pos, relative_pos, extra_info])
            
            # Nettoyage NaN/inf
            obs = np.where(np.isfinite(obs), obs, 0.0)
            obs = obs.astype(np.float32)
            
            # Padding/troncature pour dimension fixe
            expected_dim = self.observation_space.shape[0]
            if len(obs) < expected_dim:
                # Padding avec zéros
                padded_obs = np.zeros(expected_dim, dtype=np.float32)
                padded_obs[:len(obs)] = obs
                obs = padded_obs
            elif len(obs) > expected_dim:
                # Troncature
                obs = obs[:expected_dim]
            
            return obs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur observation: {e}")
            # Observation par défaut sécurisée
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _check_termination(self, positions):
        """Vérification de fin d'épisode comme le collègue"""
        
        dist = positions['palm_to_cube_dist']
        cube_pos = positions['cube_pos']
        
        # Conditions de terminaison comme le collègue
        if (dist > 0.5 or 
            cube_pos[2] < 0.01 or 
            cube_pos[2] > 1.0 or 
            self.current_step >= self.max_episode_steps):
            return True
        
        return False
    
    def render(self):
        """Rendu de l'environnement"""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data)
                return self.renderer.render()
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur rendu: {e}")
        return None
    
    def close(self):
        """Fermeture propre"""
        if hasattr(self, 'renderer'):
            try:
                self.renderer.close()
            except:
                pass
        
        # Nettoyage du fichier temporaire
        if hasattr(self, 'model_path') and self.model_path:
            try:
                Path(self.model_path).unlink(missing_ok=True)
            except:
                pass
        
        self.logger.info("🔒 Environnement fermé proprement")
    
    def advance_curriculum_level(self, episode_reward: float) -> bool:
        """Avance le niveau de curriculum si performance suffisante"""
        
        # Critères d'avancement progressifs
        thresholds = {
            1: -20.0,  # Niveau débutant
            2: -10.0,  # Niveau intermédiaire  
            3: 0.0,    # Niveau avancé
            4: 10.0,   # Niveau expert
            5: 20.0    # Niveau maître
        }
        
        if (self.curriculum_level < 5 and 
            episode_reward > thresholds.get(self.curriculum_level, 0)):
            
            self.curriculum_level += 1
            self.logger.info(f"🎓 Curriculum avancé au niveau {self.curriculum_level}")
            return True
        
        return False


def make_optimized_grasp_env(**kwargs):
    """Factory pour créer l'environnement optimisé"""
    return OptimizedGraspEnv(**kwargs)