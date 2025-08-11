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
import os

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
    
# Dans le __init__ de OptimizedGraspEnv, remplacer cette ligne :
# self.model_path = model_path or self._create_optimized_model()

# Par :
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
            
        # CORRECTION: Utiliser le modèle unique disponible
        if model_path is None:
            # Chercher le modèle g1_combined dans results/
            model_path = "/home/oussema/Documents/project/results/g1_combined.xml"
                
            # Vérifier si le modèle existe
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Modèle introuvable: {model_path}")
            # Créer un modèle minimal qui fonctionne
            model_path = self._create_minimal_working_model()
            
        self.model_path = model_path
        self.logger.info(f"🔧 Utilisation du modèle: {model_path}")
            
        try:
            # Charger le modèle MuJoCo
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
                
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation environnement: {e}")
            raise
    
    # Le reste du code reste identique...
    def _create_minimal_working_model(self) -> str:
        """Crée un modèle XML minimal mais fonctionnel"""
        
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="optimized_grasp_minimal">
        <compiler angle="radian" coordinate="local" />
        <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="20"/>
        
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" size="2 2 0.1" type="box" pos="0 0 -0.1" rgba="0.5 0.5 0.5 1"/>
            
            <!-- Table stable -->
            <body name="table" pos="0.3 0 0.4">
                <geom type="box" size="0.2 0.2 0.02" rgba="0.8 0.6 0.4 1" friction="1.0"/>
                <inertial pos="0 0 0" mass="50.0" diaginertia="1.0 1.0 0.1"/>
            </body>
            
            <!-- Cube à saisir -->
            <body name="cube" pos="0.15 0 0.44">
                <freejoint name="cube:joint"/>
                <geom name="cube_geom" type="box" size="0.02 0.02 0.02" 
                    rgba="0.2 0.8 0.2 1" friction="2.0" density="100"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
            </body>
            
            <!-- Robot simplifié et stable -->
            <body name="torso" pos="0 0 0.5">
                <geom type="box" size="0.05 0.05 0.1" rgba="0.7 0.7 0.7 1"/>
                <inertial pos="0 0 0" mass="5.0" diaginertia="0.1 0.1 0.1"/>
                
                <body name="shoulder" pos="0 -0.1 0.05">
                    <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-1.0 1.0" damping="0.5"/>
                    <joint name="shoulder_tilt" type="hinge" axis="0 1 0" range="-1.0 1.0" damping="0.5"/>
                    <geom type="capsule" size="0.02 0.05" rgba="0.6 0.6 0.6 1"/>
                    <inertial pos="0 0 -0.05" mass="0.5" diaginertia="0.01 0.01 0.01"/>
                    
                    <body name="elbow" pos="0 0 -0.1">
                        <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.0" damping="0.3"/>
                        <geom type="capsule" size="0.015 0.05" rgba="0.5 0.5 0.5 1"/>
                        <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.005 0.005 0.005"/>
                        
                        <body name="wrist" pos="0 0 -0.1">
                            <joint name="wrist_pitch" type="hinge" axis="1 0 0" range="-1.0 1.0" damping="0.2"/>
                            <geom type="capsule" size="0.01 0.03" rgba="0.4 0.4 0.4 1"/>
                            <inertial pos="0 0 -0.03" mass="0.2" diaginertia="0.002 0.002 0.002"/>
                            
                            <body name="right_hand_index_1_link" pos="0 0 -0.04">
                                <geom name="palm_geom" type="box" size="0.02 0.015 0.01" rgba="0.9 0.7 0.5 1"/>
                                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                                
                                <body name="right_hand_thumb_2_link" pos="0.015 0.01 0">
                                    <joint name="right_hand_thumb_base" type="hinge" axis="1 0 0" range="0 0.8" damping="0.1"/>
                                    <geom name="right_hand_thumb_2_geom" type="capsule" size="0.005 0.015" 
                                        rgba="0.9 0.7 0.5 1" friction="3.0"/>
                                    <inertial pos="0 0 0.008" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                                </body>
                                
                                <body name="right_hand_index_2_link" pos="0.02 0.005 0">
                                    <joint name="right_hand_index_base" type="hinge" axis="0 1 0" range="0 0.8" damping="0.1"/>
                                    <geom name="right_hand_index_1_geom" type="capsule" size="0.005 0.015" 
                                        rgba="0.9 0.7 0.5 1" friction="3.0"/>
                                    <inertial pos="0 0 0.008" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                                </body>
                                
                                <body name="right_hand_middle_1_link" pos="0.02 -0.005 0">
                                    <joint name="right_hand_middle_base" type="hinge" axis="0 1 0" range="0 0.8" damping="0.1"/>
                                    <geom name="right_hand_middle_1_geom" type="capsule" size="0.005 0.015" 
                                        rgba="0.9 0.7 0.5 1" friction="3.0"/>
                                    <inertial pos="0 0 0.008" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <position name="shoulder_pan_motor" joint="shoulder_pan" kp="10" kv="3" forcerange="-5 5"/>
            <position name="shoulder_tilt_motor" joint="shoulder_tilt" kp="10" kv="3" forcerange="-5 5"/>
            <position name="elbow_motor" joint="elbow" kp="8" kv="2" forcerange="-3 3"/>
            <position name="wrist_pitch_motor" joint="wrist_pitch" kp="5" kv="1" forcerange="-2 2"/>
            <position name="thumb_base_motor" joint="right_hand_thumb_base" kp="3" kv="0.5" forcerange="-1 1"/>
            <position name="index_base_motor" joint="right_hand_index_base" kp="3" kv="0.5" forcerange="-1 1"/>
            <position name="middle_base_motor" joint="right_hand_middle_base" kp="3" kv="0.5" forcerange="-1 1"/>
        </actuator>
    </mujoco>'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write(model_xml)
            self.logger.info(f"🔧 Modèle minimal créé: {f.name}")
            return f.name
    def _setup_logging(self):
        """Configure le logging"""
        self.logger = logging.getLogger("OptimizedGrasp")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _create_optimized_model(self) -> str:
        """Crée un modèle XML optimisé inspiré du collègue"""
        
        model_xml = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="minimal_grasp">
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" size="2 2 0.1" type="box" pos="0 0 -0.1" rgba="0.5 0.5 0.5 1"/>
        
        <!-- Table -->
        <body name="table" pos="0.5 0 0.01">
            <geom type="box" size="0.4 0.3 0.02" rgba="0.8 0.6 0.4 1"/>
        </body>
        
        <!-- Cube -->
        <body name="cube" pos="0.3 0 0.05">
            <freejoint/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" 
                  rgba="0.2 0.8 0.2 1" friction="5.0 1.0 0.5"/>
            <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
        </body>
        
        <!-- Bras simple -->
        <body name="shoulder" pos="0 0 0.5">
            <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
            <joint name="shoulder_tilt" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom type="capsule" size="0.04 0.1" rgba="0.7 0.7 0.7 1"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
            
            <body name="elbow" pos="0 0 -0.2">
                <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.5"/>
                <geom type="capsule" size="0.03 0.1" rgba="0.6 0.6 0.6 1"/>
                <inertial pos="0 0 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
                
                <body name="wrist" pos="0 0 -0.2">
                    <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="capsule" size="0.025 0.05" rgba="0.5 0.5 0.5 1"/>
                    <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
                    
                    <!-- Main simple -->
                    <body name="right_hand_index_1_link" pos="0 0 -0.08">
                        <geom type="box" size="0.03 0.04 0.02" rgba="0.9 0.7 0.5 1"/>
                        <inertial pos="0 0 0" mass="0.3" diaginertia="0.003 0.003 0.003"/>
                        
                        <!-- Doigts simples -->
                        <body name="right_hand_thumb_2_link" pos="0.02 0.03 0">
                            <joint name="right_hand_thumb_base" type="hinge" axis="1 0 0" range="0 1.2"/>
                            <geom name="right_hand_thumb_2_geom" type="capsule" size="0.008 0.02" 
                                  rgba="0.9 0.7 0.5 1" friction="2.0"/>
                            <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
                        </body>
                        
                        <body name="right_hand_index_2_link" pos="0.04 0.01 0">
                            <joint name="right_hand_index_base" type="hinge" axis="0 1 0" range="0 1.2"/>
                            <geom name="right_hand_index_1_geom" type="capsule" size="0.008 0.025" 
                                  rgba="0.9 0.7 0.5 1" friction="2.0"/>
                            <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
                        </body>
                        
                        <body name="right_hand_middle_1_link" pos="0.04 -0.01 0">
                            <joint name="right_hand_middle_base" type="hinge" axis="0 1 0" range="0 1.2"/>
                            <geom name="right_hand_middle_1_geom" type="capsule" size="0.008 0.025" 
                                  rgba="0.9 0.7 0.5 1" friction="2.0"/>
                            <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="shoulder_pan_motor" joint="shoulder_pan" kp="15" kv="5"/>
        <position name="shoulder_tilt_motor" joint="shoulder_tilt" kp="15" kv="5"/>
        <position name="elbow_motor" joint="elbow" kp="12" kv="4"/>
        <position name="wrist_pitch_motor" joint="wrist_pitch" kp="10" kv="3"/>
        <position name="thumb_base_motor" joint="right_hand_thumb_base" kp="8" kv="2"/>
        <position name="index_base_motor" joint="right_hand_index_base" kp="8" kv="2"/>
        <position name="middle_base_motor" joint="right_hand_middle_base" kp="8" kv="2"/>
    </actuator>
</mujoco>'''
        
        # Sauvegarder le modèle
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(model_xml)
            return f.name
    
# Dans envs/optimized_grasp_env.py, modifiez la méthode _load_mujoco_model :

    def _load_mujoco_model(self):
        """Charge le modèle MuJoCo avec gestion d'erreurs robuste"""
        try:
            # SOLUTION 1: Vérifier et nettoyer le fichier XML avant chargement
            if os.path.exists(self.model_path):
                # Lire le contenu du fichier pour diagnostiquer
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Vérifier si le contenu est valide
                if not content.strip():
                    raise Exception("Fichier XML vide")
                
                if not content.strip().startswith('<?xml'):
                    raise Exception("Fichier XML mal formaté (pas de déclaration XML)")
                
                # Log des premières lignes pour debug
                lines = content.split('\n')[:5]
                self.logger.info(f"📄 Premières lignes du XML: {lines}")
                
                # Essayer de charger normalement
                try:
                    self.model = mujoco.MjModel.from_xml_path(self.model_path)
                    self.data = mujoco.MjData(self.model)
                    self.logger.info(f"✅ Modèle MuJoCo chargé: {self.model.nq} DOFs, {self.model.nu} actuateurs")
                    
                except Exception as xml_error:
                    self.logger.warning(f"⚠️ Erreur XML avec fichier existant: {xml_error}")
                    # Fallback vers modèle minimal
                    self.logger.info("🔧 Création d'un modèle minimal de secours...")
                    self.model_path = self._create_minimal_working_model()
                    self.model = mujoco.MjModel.from_xml_path(self.model_path)
                    self.data = mujoco.MjData(self.model)
                    self.logger.info("✅ Modèle minimal chargé avec succès")
            else:
                # Fichier n'existe pas
                self.logger.warning(f"⚠️ Fichier non trouvé: {self.model_path}")
                self.model_path = self._create_minimal_working_model()
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
                self.data = mujoco.MjData(self.model)
                self.logger.info("✅ Modèle minimal créé et chargé")
            
            # Configuration du rendu
            if self.render_mode == "rgb_array":
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            raise

    def _resolve_model_path(self, model_path):
        """Résout intelligemment le chemin du modèle avec fallback immédiat"""
        
        # MODIFICATION: Ne jamais utiliser le modèle problématique
        # Créer directement un modèle minimal qui fonctionne
        self.logger.info("🔧 Création directe d'un modèle minimal (bypass du modèle problématique)")
        return None  # Forcer la création d'un modèle minimal
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
        """Reset avec positions initiales aléatoires"""
        
        obs, info = super().reset(seed=seed, options=options)
        
        # RANDOMISER POSITION INITIALE pour éviter blocage local
        try:
            # Position cube légèrement aléatoire autour de [0.18, 0.0, 0.04]
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            if cube_joint_id >= 0:
                cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
                
                # Position avec variation aléatoire
                base_pos = np.array([0.18, 0.0, 0.04])
                random_offset = np.random.normal(0, 0.02, 3)  # Petite variation
                random_pos = base_pos + random_offset
                
                # Appliquer position
                start = cube_qpos_addr
                end = min(cube_qpos_addr + 3, len(self.data.qpos))
                self.data.qpos[start:end] = random_pos[:end-start]
            
            # RANDOMISER POSITION ROBOT aussi
            if len(self.data.qpos) > 7:  # Si on a des joints de robot
                for i in range(min(4, len(self.data.qpos) - 7)):  # Premiers joints
                    self.data.qpos[7 + i] += np.random.normal(0, 0.3)  # Variation importante
                    
        except Exception as e:
            self.logger.warning(f"Randomisation position échouée: {e}")
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
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
            # CORRECTION: Créer les poids selon la taille actuelle de l'historique
            history_size = len(self.action_history)
            
            # Moyenne pondérée simple : plus de poids sur les actions récentes
            smoothed = np.zeros_like(action)
            total_weight = 0
            
            # Créer des poids croissants pour les actions plus récentes
            for i, hist_action in enumerate(self.action_history):
                weight = i + 1  # Poids croissant (1, 2, 3, 4, 5)
                smoothed += weight * hist_action
                total_weight += weight
            
            # Normaliser
            smoothed /= total_weight
            return smoothed
        
        return action
    def _get_positions(self):
            """Calcule toutes les positions nécessaires"""
            
            try:
                # Positions des objets
                cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
                cube_pos = self.data.xpos[cube_id] if cube_id >= 0 else np.zeros(3)
                
                # Position de la main
                try:
                    palm_pos = self.data.body("right_hand_index_1_link").xpos
                except:
                    palm_pos = np.array([0.0, 0.0, 0.5])  # Position par défaut
                
                # Positions des doigts
                finger_positions = {}
                finger_names = ["right_hand_thumb_2_link", "right_hand_index_2_link", "right_hand_middle_1_link"]
                
                for name in finger_names:
                    try:
                        finger_positions[name] = self.data.body(name).xpos
                    except:
                        finger_positions[name] = palm_pos  # Fallback
                
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
        
    def _count_finger_contacts(self):
            """Compte les contacts des doigts avec le cube (comme le collègue)"""
            
            contact_count = 0
            finger_geoms = ["right_hand_thumb_2_geom", "right_hand_index_1_geom", "right_hand_middle_1_geom"]
            
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                try:
                    name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    
                    # Vérifier si c'est un contact doigt-cube
                    if ((name1 == "cube_geom" and name2 in finger_geoms) or
                        (name2 == "cube_geom" and name1 in finger_geoms)):
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
        """Récompense agressive pour débloquer l'apprentissage"""
        
        dist = positions['palm_to_cube_dist']
        cube_vel = positions['cube_velocity']
        contact_count = positions['contact_count']
        
        reward = 0.0
        
        # RÉCOMPENSE DISTANCE TRÈS AGRESSIVE
        if dist < 0.03:
            reward += 200.0  # Contact imminent
        elif dist < 0.05:
            reward += 100.0  # Très proche
        elif dist < 0.08:
            reward += 50.0   # Proche
        elif dist < 0.12:
            reward += 20.0   # Assez proche  
        elif dist < 0.18:
            reward += 5.0    # Approche
        else:
            reward -= 20.0   # Trop loin = pénalité

        # BONUS CONTACT ÉNORME
        if contact_count >= 1:
            reward += 500.0    # Premier contact = énorme bonus!
        if contact_count >= 2:
            reward += 1000.0   # Grasp partiel = jackpot!
        if contact_count >= 3:
            reward += 2000.0   # Grasp complet = méga jackpot!

        # Bonus amélioration
        if dist < self.best_distance:
            improvement = self.best_distance - dist
            reward += improvement * 200.0  # x200 pour forcer apprentissage
            self.best_distance = dist

        # Pénalité vitesse réduite
        reward -= min(10.0, cube_vel * 5.0)

        # Pénalité temps très faible
        reward -= 0.01

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
        if self.render_mode == "rgb_array" and hasattr(self, 'renderer'):
            self.renderer.update_scene(self.data)
            return self.renderer.render()
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
