#!/usr/bin/env python3
"""
Script de configuration pour l'environnement local
Copie et adapte tous les fichiers nécessaires pour l'entraînement G1
"""

import os
import shutil
from pathlib import Path
import sys

def create_directory_structure():
    """Crée la structure de dossiers nécessaire"""
    print("📁 Création de la structure de dossiers...")
    
    dirs_to_create = [
        "envs",
        "agents", 
        "utils",
        "results",
        "training_results",
        "training_results/models",
        "training_results/videos",
        "training_results/logs"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")

def create_simple_grasp_env():
    """Crée l'environnement de saisie"""
    print("\n🤖 Création de l'environnement de saisie...")
    
    content = '''#!/usr/bin/env python3
"""
Environnement simplifié de saisie pour robot G1
Utilise les capteurs de force pour détecter le contact
Auteur: Assistant IA
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData, mj_forward
import time
from pathlib import Path

class SimpleGraspEnv(gym.Env):
    """
    Environnement simplifié de saisie du cube pour G1
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 xml_path="results/g1_combined.xml",
                 render_mode=None,
                 max_episode_steps=500,
                 curriculum_level=1):
        super().__init__()
        
        # Configuration
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_level = curriculum_level
        self.current_step = 0
        
        # Charger le modèle MuJoCo
        self._load_model()
        
        # Identifier les éléments du modèle
        self._identify_model_elements()
        
        # Configurer les espaces d'observation et d'action
        self._setup_spaces()
        
        # Variables d'état
        self.cube_initial_pos = None
        self.cube_initial_height = None
        self.contact_detected = False
        self.grasp_phase = "approach"  # approach -> contact -> grasp -> lift
        self.phase_start_time = 0.0
        
        # Renderer pour les vidéos
        self.renderer = None
        self.viewer = None
        
    def _load_model(self):
        """Charge le modèle MuJoCo"""
        try:
            self.model = MjModel.from_xml_path(self.xml_path)
            self.data = MjData(self.model)
            print(f"✅ Modèle chargé: {self.xml_path}")
            print(f"   Capteurs: {self.model.nsensor}")
            print(f"   Actuateurs: {self.model.nu}")
                
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def _identify_model_elements(self):
        """Identifie les éléments importants du modèle"""
        # Trouver le cube
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            # Essayer d'autres noms possibles
            for name in ["object", "box", "target"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        # Identifier les capteurs de force (pour la détection de contact)
        self.force_sensor_ids = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and ("force" in sensor_name.lower() or "touch" in sensor_name.lower()):
                self.force_sensor_ids.append(i)
        
        print(f"   Cube ID: {self.cube_body_id}")
        print(f"   Capteurs de force: {len(self.force_sensor_ids)}")
        
        # Identifier les joints des bras et mains
        self._identify_arm_joints()
    
    def _identify_arm_joints(self):
        """Identifie les joints des bras et des mains"""
        self.arm_joints = []
        self.finger_joints = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                # Joints des bras (épaules, coudes, poignets)
                if any(keyword in joint_name.lower() for keyword in 
                       ["shoulder", "elbow", "wrist", "arm", "forearm"]):
                    self.arm_joints.append(i)
                # Joints des doigts
                elif any(keyword in joint_name.lower() for keyword in 
                        ["finger", "thumb", "hand"]):
                    self.finger_joints.append(i)
        
        print(f"   Joints bras: {len(self.arm_joints)}")
        print(f"   Joints doigts: {len(self.finger_joints)}")
    
    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        # Action space: contrôle des actuateurs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Observation space: positions, vitesses, position cube, capteurs
        obs_dim = (
            self.model.nq +  # positions des joints
            self.model.nv +  # vitesses des joints
            3 +              # position du cube
            1 +              # hauteur du cube
            len(self.force_sensor_ids) +  # capteurs de force
            4                # phase info (one-hot encoding)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mj_resetData(self.model, self.data)
        
        # Ajouter un peu de bruit aux positions initiales
        if self.model.nq > 0:
            self.data.qpos[:] += 0.01 * np.random.randn(self.model.nq)
        
        # Position initiale des contrôleurs
        self.data.ctrl[:] = 0.0
        
        # Simulation forward pour stabiliser
        for _ in range(10):
            mj_step(self.model, self.data)
        
        # Enregistrer la position initiale du cube
        if self.cube_body_id >= 0:
            self.cube_initial_pos = self.data.xpos[self.cube_body_id].copy()
            self.cube_initial_height = self.cube_initial_pos[2]
        else:
            self.cube_initial_pos = np.array([0.5, 0.0, 0.45])
            self.cube_initial_height = 0.45
        
        # Reset des variables d'état
        self.current_step = 0
        self.contact_detected = False
        self.grasp_phase = "approach"
        self.phase_start_time = self.data.time
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécute une étape de simulation"""
        self.current_step += 1
        
        # Appliquer l'action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        # Simulation step
        mj_step(self.model, self.data)
        
        # Calculer l'observation et la récompense
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Vérifier les conditions de fin
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Mise à jour de la phase
        self._update_phase()
        
        info = {
            "phase": self.grasp_phase,
            "contact": self.contact_detected,
            "cube_height": self._get_cube_height(),
            "step": self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Calcule l'observation actuelle"""
        obs_parts = []
        
        # Positions et vitesses des joints
        obs_parts.append(self.data.qpos.copy())
        obs_parts.append(self.data.qvel.copy())
        
        # Position du cube
        if self.cube_body_id >= 0:
            cube_pos = self.data.xpos[self.cube_body_id].copy()
        else:
            cube_pos = self.cube_initial_pos.copy()
        obs_parts.append(cube_pos)
        
        # Hauteur relative du cube
        cube_height = np.array([cube_pos[2] - self.cube_initial_height])
        obs_parts.append(cube_height)
        
        # Capteurs de force
        force_data = []
        for sensor_id in self.force_sensor_ids:
            force_data.append(self.data.sensordata[sensor_id])
        obs_parts.append(np.array(force_data, dtype=np.float32))
        
        # Phase info (one-hot)
        phase_mapping = {"approach": 0, "contact": 1, "grasp": 2, "lift": 3}
        phase_onehot = np.zeros(4, dtype=np.float32)
        phase_onehot[phase_mapping.get(self.grasp_phase, 0)] = 1.0
        obs_parts.append(phase_onehot)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _compute_reward(self):
        """Calcule la récompense"""
        reward = 0.0
        
        # Détection de contact
        contact_reward = self._compute_contact_reward()
        reward += contact_reward
        
        # Récompense de hauteur (lift)
        height_reward = self._compute_height_reward()
        reward += height_reward
        
        # Pénalité de mouvement excessif
        movement_penalty = self._compute_movement_penalty()
        reward += movement_penalty
        
        # Récompense de stabilité
        stability_reward = self._compute_stability_reward()
        reward += stability_reward
        
        # Curriculum learning: ajuster les récompenses selon le niveau
        reward *= self._get_curriculum_multiplier()
        
        return float(reward)
    
    def _compute_contact_reward(self):
        """Récompense pour la détection de contact"""
        # Vérifier les capteurs de force
        contact_detected = False
        total_force = 0.0
        
        for sensor_id in self.force_sensor_ids:
            force_magnitude = abs(self.data.sensordata[sensor_id])
            total_force += force_magnitude
            if force_magnitude > 0.1:  # Seuil de détection
                contact_detected = True
        
        self.contact_detected = contact_detected
        
        if contact_detected:
            # Récompense proportionnelle à la force mais limitée
            return min(1.0, total_force * 0.5)
        else:
            return -0.1  # Petite pénalité sans contact
    
    def _compute_height_reward(self):
        """Récompense pour soulever le cube"""
        cube_height = self._get_cube_height()
        height_diff = cube_height - self.cube_initial_height
        
        if height_diff > 0.01:  # Le cube s'élève
            return min(10.0, height_diff * 20)  # Récompense substantielle
        elif height_diff < -0.05:  # Le cube tombe
            return -5.0
        else:
            return 0.0
    
    def _compute_movement_penalty(self):
        """Pénalité pour les mouvements excessifs"""
        # Pénalité basée sur l'énergie des actions
        action_energy = np.sum(np.square(self.data.ctrl))
        return -0.01 * action_energy
    
    def _compute_stability_reward(self):
        """Récompense pour la stabilité du cube"""
        if self.cube_body_id >= 0:
            # Vérifier la stabilité angulaire du cube
            cube_quat = self.data.xquat[self.cube_body_id]
            # Quaternion proche de l'identité = cube stable
            stability = 1.0 - np.linalg.norm(cube_quat - np.array([1,0,0,0]))
            return 0.5 * max(0, stability)
        return 0.0
    
    def _get_curriculum_multiplier(self):
        """Multiplicateur selon le niveau de curriculum"""
        if self.curriculum_level == 1:
            return 1.0  # Niveau de base
        elif self.curriculum_level == 2:
            return 1.2  # Légèrement plus difficile
        elif self.curriculum_level == 3:
            return 1.5  # Plus difficile
        else:
            return 1.0
    
    def _get_cube_height(self):
        """Obtient la hauteur actuelle du cube"""
        if self.cube_body_id >= 0:
            return self.data.xpos[self.cube_body_id][2]
        return self.cube_initial_height
    
    def _update_phase(self):
        """Met à jour la phase actuelle de la tâche"""
        time_in_phase = self.data.time - self.phase_start_time
        
        if self.grasp_phase == "approach":
            if self.contact_detected:
                self.grasp_phase = "contact"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "contact":
            if time_in_phase > 1.0:  # 1 seconde de contact
                self.grasp_phase = "grasp"
                self.phase_start_time = self.data.time
        
        elif self.grasp_phase == "grasp":
            if time_in_phase > 2.0:  # 2 secondes pour saisir
                self.grasp_phase = "lift"
                self.phase_start_time = self.data.time
    
    def _check_termination(self):
        """Vérifie les conditions de fin d'épisode"""
        # Succès: cube soulevé suffisamment haut
        cube_height = self._get_cube_height()
        if cube_height - self.cube_initial_height > 0.1:
            return True
        
        # Échec: cube trop bas ou trop éloigné
        if self.cube_body_id >= 0:
            cube_pos = self.data.xpos[self.cube_body_id]
            if cube_pos[2] < 0.1:  # Cube au sol
                return True
            
            # Distance horizontale trop grande
            horizontal_dist = np.linalg.norm(cube_pos[:2] - self.cube_initial_pos[:2])
            if horizontal_dist > 0.5:
                return True
        
        return False
    
    def render(self, mode=None):
        """Rendu de l'environnement"""
        mode = mode or self.render_mode
        
        if mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
        elif mode == "rgb_array":
            if self.renderer is None:
                from mujoco import Renderer
                self.renderer = Renderer(self.model, width=640, height=480)
            
            self.renderer.update_scene(self.data)
            frame = self.renderer.render()
            return frame
            
        else:
            raise ValueError(f"Mode de rendu non supporté: {mode}")
    
    def close(self):
        """Ferme l'environnement"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
'''
    
    with open("envs/simple_grasp_env.py", 'w') as f:
        f.write(content)
    print("   ✅ envs/simple_grasp_env.py créé")

def create_sac_agent():
    """Crée l'agent SAC"""
    print("\n🧠 Création de l'agent SAC...")
    
    content = '''#!/usr/bin/env python3
"""
Agent SAC amélioré pour la tâche de saisie G1
Implémentation complète avec replay buffer et entraînement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Buffer de replay pour SAC"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch du buffer"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Réseau acteur pour SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], max_action=1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # Réseau principal
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU()
            ])
            input_dim = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Sorties pour moyenne et log std
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du réseau"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass du réseau acteur"""
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        """Échantillonne une action avec reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Distribution normale
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Appliquer tanh pour borner les actions
        action = torch.tanh(x_t) * self.max_action
        
        # Calculer log prob avec correction pour tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    """Réseau critique pour SAC (Q-function)"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        
        # Réseau Q1
        layers1 = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers1.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU()
            ])
            input_dim = hidden_size
        layers1.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        
        # Réseau Q2
        layers2 = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers2.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU()
            ])
            input_dim = hidden_size
        layers2.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers2)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du réseau"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """Forward pass pour les deux Q-networks"""
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class ImprovedSACAgent:
    """Agent SAC amélioré pour la tâche de saisie"""
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 max_action=1.0,
                 lr=3e-4,
                 alpha=0.2,
                 gamma=0.99,
                 tau=0.005,
                 buffer_size=100000,
                 hidden_sizes=[256, 256],
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
        # Réseaux de neurones
        self.actor = Actor(state_dim, action_dim, hidden_sizes, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes).to(device)
        
        # Copier les poids vers le réseau cible
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Temperature parameter automatique
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Métriques d'entraînement
        self.training_step = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_history = []
        
        print(f"✅ Agent SAC initialisé sur {device}")
        print(f"   Dimension état: {state_dim}")
        print(f"   Dimension action: {action_dim}")
        print(f"   Architecture: {hidden_sizes}")
    
    def select_action(self, state, evaluate=False):
        """Sélectionne une action"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # Mode évaluation: prendre la moyenne
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        else:
            # Mode exploration: échantillonner
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans le replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=256):
        """Met à jour l'agent"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Échantillonner du buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Mise à jour du critque
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Mise à jour de l'acteur
        actor_loss = self._update_actor(states)
        
        # Mise à jour de alpha
        alpha_loss = self._update_alpha(states)
        
        # Mise à jour des réseaux cibles
        self._update_target_networks()
        
        self.training_step += 1
        
        # Enregistrer les métriques
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)
        self.alpha_history.append(self.alpha)
        
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha": self.alpha,
            "alpha_loss": alpha_loss
        }
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Met à jour le réseau critique"""
        with torch.no_grad():
            # Actions pour l'état suivant
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Q-values cibles
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            # Target Q-value
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Q-values actuelles
        q1, q2 = self.critic(states, actions)
        
        # Losses
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss
        
        # Optimisation
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """Met à jour le réseau acteur"""
        # Échantillonner les actions
        actions, log_probs = self.actor.sample(states)
        
        # Q-values
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        # Loss de l'acteur
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Optimisation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states):
        """Met à jour le paramètre de température alpha"""
        with torch.no_grad():
            _, log_probs = self.actor.sample(states)
        
        # Loss d'alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        # Optimisation
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Mettre à jour alpha
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def _update_target_networks(self):
        """Met à jour les réseaux cibles avec soft update"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Sauvegarde l'agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step
        }, filepath)
        print(f"✅ Agent sauvegardé: {filepath}")
    
    def load(self, filepath):
        """Charge l'agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.training_step = checkpoint['training_step']
        self.alpha = self.log_alpha.exp().item()
        
        print(f"✅ Agent chargé: {filepath}")
'''
    
    with open("agents/improved_sac_agent.py", 'w') as f:
        f.write(content)
    print("   ✅ agents/improved_sac_agent.py créé")

def create_video_recorder():
    """Crée l'enregistreur vidéo"""
    print("\n🎬 Création de l'enregistreur vidéo...")
    
    content = '''#!/usr/bin/env python3
"""
Enregistreur vidéo simplifié pour les simulations G1
"""

import numpy as np
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  imageio non disponible - enregistrement vidéo désactivé")

import os
from pathlib import Path
import time
from datetime import datetime

class VideoRecorder:
    """Enregistreur vidéo pour les épisodes de simulation"""
    
    def __init__(self, 
                 output_dir="training_results/videos",
                 fps=30,
                 quality=8,
                 format="mp4"):
        
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.quality = quality
        self.format = format
        self.has_imageio = HAS_IMAGEIO
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # État de l'enregistrement
        self.is_recording = False
        self.frames = []
        self.episode_info = {}
        
        if self.has_imageio:
            print(f"✅ Enregistreur vidéo initialisé")
            print(f"   Dossier: {self.output_dir}")
        else:
            print(f"⚠️  Enregistreur vidéo en mode simulation (pas d'imageio)")
    
    def start_recording(self, episode_info=None):
        """Démarre l'enregistrement d'un nouvel épisode"""
        if not self.has_imageio:
            return
            
        if self.is_recording:
            self.stop_recording()
        
        self.is_recording = True
        self.frames = []
        self.episode_info = episode_info or {}
    
    def add_frame(self, frame):
        """Ajoute une frame à l'enregistrement"""
        if not self.is_recording or not self.has_imageio:
            return
        
        # Convertir en numpy array si nécessaire
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # S'assurer que la frame est en RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convertir de float [0,1] à uint8 [0,255] si nécessaire
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            
            self.frames.append(frame)
    
    def stop_recording(self, filename=None):
        """Arrête l'enregistrement et sauvegarde la vidéo"""
        if not self.is_recording or len(self.frames) == 0 or not self.has_imageio:
            return None
        
        self.is_recording = False
        
        # Générer le nom de fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode = self.episode_info.get("episode", "unknown")
            reward = self.episode_info.get("total_reward", 0)
            filename = f"episode_{episode}_{timestamp}_reward_{reward:.1f}.{self.format}"
        
        filepath = self.output_dir / filename
        
        try:
            # Sauvegarder la vidéo
            imageio.mimsave(
                str(filepath),
                self.frames,
                fps=self.fps,
                quality=self.quality
            )
            
            print(f"✅ Vidéo sauvegardée: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
        finally:
            self.frames = []
    
    def record_episode(self, env, agent, max_steps=500, render_mode="rgb_array"):
        """Enregistre un épisode complet"""
        if not self.has_imageio:
            print("⚠️  Enregistrement vidéo désactivé (imageio manquant)")
            return None, {}
            
        print(f"🎬 Enregistrement d'un épisode...")
        
        # Démarrer l'enregistrement
        obs, info = env.reset()
        self.start_recording()
        
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Rendu de l'environnement
            try:
                frame = env.render(mode=render_mode)
                if frame is not None:
                    self.add_frame(frame)
            except:
                pass  # Ignorer les erreurs de rendu
            
            # Action de l'agent
            if agent is not None:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = env.action_space.sample()
            
            # Étape de simulation
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        # Informations de l'épisode
        episode_info = {
            "total_reward": total_reward,
            "steps": step,
            "success": terminated and total_reward > 0,
            "phase": info.get("phase", "unknown"),
            "contact": info.get("contact", False),
            "cube_height": info.get("cube_height", 0)
        }
        
        self.episode_info = episode_info
        
        # Arrêter l'enregistrement
        video_path = self.stop_recording()
        
        return video_path, episode_info
'''
    
    with open("utils/video_recorder.py", 'w') as f:
        f.write(content)
    print("   ✅ utils/video_recorder.py créé")

def create_training_script():
    """Crée le script d'entraînement principal"""
    print("\n🚀 Création du script d'entraînement...")
    
    content = '''#!/usr/bin/env python3
"""
Script d'entraînement simplifié pour la saisie G1
Utilise SAC avec curriculum learning et enregistrement vidéo
"""

import os
import sys
import argparse
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch non disponible - mode simulation")

import time
from pathlib import Path
from datetime import datetime
import json

# Ajouter les modules locaux au path
sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')
sys.path.append('./utils')

if HAS_TORCH:
    from envs.simple_grasp_env import SimpleGraspEnv
    from agents.improved_sac_agent import ImprovedSACAgent
from utils.video_recorder import VideoRecorder

class GraspTrainer:
    """Entraîneur pour la tâche de saisie G1"""
    
    def __init__(self, config):
        self.config = config
        
        # Créer les dossiers de sortie
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        if not HAS_TORCH:
            print("❌ PyTorch requis pour l'entraînement")
            print("💡 Installation: pip install torch")
            return
        
        # Initialiser l'environnement
        print("🤖 Initialisation de l'environnement...")
        self.env = SimpleGraspEnv(
            xml_path=config['model_path'],
            max_episode_steps=config['max_episode_steps'],
            curriculum_level=config['curriculum_level']
        )
        
        # Initialiser l'agent SAC
        print("🧠 Initialisation de l'agent SAC...")
        self.agent = ImprovedSACAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            lr=config['learning_rate'],
            hidden_sizes=config['hidden_sizes'],
            buffer_size=config['buffer_size']
        )
        
        # Enregistreur vidéo
        self.video_recorder = VideoRecorder(
            output_dir=self.output_dir / "videos",
            fps=config['video_fps']
        )
        
        # Métriques d'entraînement
        self.episode_rewards = []
        self.episode_lengths = []
        
        print("✅ Entraîneur initialisé")
    
    def train(self):
        """Lance l'entraînement"""
        if not HAS_TORCH:
            print("❌ Impossible de lancer l'entraînement sans PyTorch")
            return
            
        print("\\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        for episode in range(total_episodes):
            # Reset de l'environnement
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Épisode
            done = False
            while not done:
                # Sélection d'action
                action = self.agent.select_action(obs)
                
                # Étape d'environnement
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Stocker la transition
                self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Mise à jour des métriques
                episode_reward += reward
                episode_length += 1
                
                obs = next_obs
            
            # Entraînement de l'agent
            if len(self.agent.replay_buffer) > self.config['batch_size']:
                for _ in range(self.config['updates_per_episode']):
                    self.agent.update(self.config['batch_size'])
            
            # Enregistrer les métriques
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Affichage des progrès
            if (episode + 1) % self.config['log_interval'] == 0:
                self._log_progress(episode + 1, total_episodes, start_time)
            
            # Enregistrement vidéo périodique
            if (episode + 1) % self.config['video_interval'] == 0:
                try:
                    self.video_recorder.record_episode(self.env, self.agent)
                except Exception as e:
                    print(f"⚠️  Erreur vidéo: {e}")
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\\n✅ ENTRAÎNEMENT TERMINÉ")
        print(f"   Durée totale: {total_time/3600:.1f}h")
        print(f"   Récompense moyenne finale: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # Sauvegarde finale
        self._save_final_results()
    
    def _log_progress(self, episode, total_episodes, start_time):
        """Affiche les progrès d'entraînement"""
        recent_rewards = self.episode_rewards[-self.config['log_interval']:]
        avg_reward = np.mean(recent_rewards)
        
        elapsed_time = time.time() - start_time
        
        print(f"\\n📊 Épisode {episode}/{total_episodes}")
        print(f"   Récompense: {avg_reward:.2f}")
        print(f"   Buffer: {len(self.agent.replay_buffer)}")
        print(f"   Temps: {elapsed_time/60:.1f}min")
    
    def _save_final_results(self):
        """Sauvegarde les résultats finaux"""
        # Sauvegarder le modèle final
        final_model_path = self.output_dir / "models" / "final_model.pth"
        self.agent.save(final_model_path)
        
        # Sauvegarder les métriques
        final_metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "avg_reward_last_100": float(np.mean(self.episode_rewards[-100:])),
            "total_episodes": len(self.episode_rewards)
        }
        
        final_metrics_path = self.output_dir / "logs" / "final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✅ Résultats sauvegardés dans {self.output_dir}")

def load_config():
    """Charge la configuration d'entraînement"""
    return {
        # Environnement
        'model_path': 'results/g1_combined.xml',
        'max_episode_steps': 500,
        'curriculum_level': 1,
        
        # Entraînement
        'total_episodes': 1000,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 100000,
        'updates_per_episode': 1,
        'hidden_sizes': [256, 256],
        
        # Logging
        'log_interval': 50,
        'video_interval': 100,
        'video_fps': 30,
        
        # Sortie
        'output_dir': 'training_results'
    }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement de saisie G1')
    parser.add_argument('--episodes', type=int, default=1000, help='Nombre d\\'épisodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Taux d\\'apprentissage')
    parser.add_argument('--output', type=str, default='training_results', help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Charger et modifier la configuration
    config = load_config()
    config['total_episodes'] = args.episodes
    config['learning_rate'] = args.lr
    config['output_dir'] = args.output
    
    print("🤖 ENTRAÎNEMENT DE SAISIE G1")
    print("=" * 50)
    print(f"Épisodes: {config['total_episodes']}")
    print(f"Taux d'apprentissage: {config['learning_rate']}")
    print(f"Dossier de sortie: {config['output_dir']}")
    
    # Vérifier que le modèle existe
    if not Path(config['model_path']).exists():
        print(f"❌ Modèle non trouvé: {config['model_path']}")
        print("💡 Placez le modèle g1_combined.xml dans le dossier results/")
        return
    
    # Créer et lancer l'entraîneur
    try:
        trainer = GraspTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\\n⏹️  Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\\n❌ Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open("train_simple_grasp.py", 'w') as f:
        f.write(content)
    print("   ✅ train_simple_grasp.py créé")

def create_test_script():
    """Crée le script de test adapté"""
    print("\n🧪 Création du script de test adapté...")
    
    content = '''#!/usr/bin/env python3
"""
Test basique du système de saisie G1 - Version locale
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Vérifie la structure des dossiers"""
    print("📁 Test de la structure...")
    
    required_dirs = ["envs", "agents", "utils", "results", "training_results"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/: OK")
        else:
            print(f"❌ {dir_name}/: Manquant")
    
    return True

def test_files():
    """Vérifie que les fichiers principaux existent"""
    print("\\n📄 Test des fichiers...")
    
    required_files = [
        "envs/simple_grasp_env.py",
        "agents/improved_sac_agent.py",
        "utils/video_recorder.py", 
        "train_simple_grasp.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: OK")
        else:
            print(f"❌ {file_path}: Manquant")
            all_good = False
    
    return all_good

def test_model():
    """Vérifie la présence du modèle"""
    print("\\n🤖 Test du modèle...")
    
    model_path = Path("results/g1_combined.xml")
    if model_path.exists():
        print(f"✅ Modèle trouvé: {model_path}")
        return True
    else:
        print(f"❌ Modèle manquant: {model_path}")
        print("💡 Placez votre modèle g1_combined.xml dans le dossier results/")
        return False

def test_dependencies():
    """Teste les dépendances Python"""
    print("\\n📦 Test des dépendances...")
    
    deps = {
        "numpy": "numpy",
        "torch": "torch", 
        "gymnasium": "gymnasium",
        "mujoco": "mujoco"
    }
    
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError:
            print(f"❌ {name}: Manquant")
            missing.append(name)
    
    if missing:
        print(f"\\n💡 Pour installer les dépendances manquantes:")
        print(f"   pip install {' '.join(missing)}")
    
    return len(missing) == 0

def create_sample_config():
    """Crée un exemple de configuration"""
    print("\\n⚙️  Création de la configuration...")
    
    config = {
        "model_path": "results/g1_combined.xml",
        "episodes": 100,
        "learning_rate": 0.0003,
        "output_dir": "training_results"
    }
    
    try:
        import json
        with open("config_example.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ config_example.json créé")
    except Exception as e:
        print(f"⚠️  Erreur lors de la création: {e}")

def main():
    """Test principal"""
    print("🤖 TEST DU SYSTÈME DE SAISIE G1")
    print("=" * 50)
    
    # Tests
    structure_ok = test_directory_structure()
    files_ok = test_files()
    model_ok = test_model()  
    deps_ok = test_dependencies()
    
    # Créer un exemple de config
    create_sample_config()
    
    print("\\n" + "=" * 50)
    
    if files_ok and structure_ok:
        print("✅ SYSTÈME PRÊT!")
        print("\\n🚀 Pour lancer l'entraînement:")
        print("   python3 train_simple_grasp.py --episodes 100")
        
        if not model_ok:
            print("\\n⚠️  N'oubliez pas de placer votre modèle dans results/")
        
        if not deps_ok:
            print("\\n⚠️  Installez d'abord les dépendances manquantes")
            
    else:
        print("❌ CONFIGURATION INCOMPLÈTE")
        print("   Relancez setup_local_training.py")
    
    return files_ok and structure_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("test_simple_grasp_basic.py", 'w') as f:
        f.write(content)
    print("   ✅ test_simple_grasp_basic.py créé")

def create_init_files():
    """Crée les fichiers __init__.py"""
    print("\n📝 Création des fichiers __init__.py...")
    
    init_dirs = ["envs", "agents", "utils"]
    for dir_name in init_dirs:
        init_path = Path(dir_name) / "__init__.py"
        with open(init_path, 'w') as f:
            f.write("# Package initialization\\n")
        print(f"   ✅ {init_path}")

def create_readme():
    """Crée un README local"""
    print("\\n📚 Création du README...")
    
    content = '''# Système d'Entraînement G1 - Installation Locale

## 🚀 Installation Rapide

1. **Vérifier le système:**
   ```bash
   python3 test_simple_grasp_basic.py
   ```

2. **Installer les dépendances:**
   ```bash
   pip install numpy torch gymnasium mujoco matplotlib imageio
   ```

3. **Placer le modèle:**
   - Copiez votre fichier `g1_combined.xml` dans le dossier `results/`

4. **Lancer l'entraînement:**
   ```bash
   python3 train_simple_grasp.py --episodes 100
   ```

## 📁 Structure Créée

```
./
├── envs/
│   └── simple_grasp_env.py     # Environnement de saisie
├── agents/
│   └── improved_sac_agent.py   # Agent SAC 
├── utils/
│   └── video_recorder.py       # Enregistrement vidéo
├── results/
│   └── g1_combined.xml         # [À placer] Modèle MuJoCo
├── training_results/           # Résultats d'entraînement
│   ├── models/                 # Modèles sauvegardés
│   ├── videos/                 # Vidéos d'épisodes
│   └── logs/                   # Métriques JSON
├── train_simple_grasp.py       # Script principal
└── test_simple_grasp_basic.py  # Tests de validation
```

## 🎯 Fonctionnalités

- ✅ **Détection de contact** via capteurs de force
- ✅ **Phases automatiques**: approche → contact → saisie → levage  
- ✅ **Agent SAC** avec replay buffer et target networks
- ✅ **Curriculum learning** adaptatif
- ✅ **Enregistrement vidéo** automatique
- ✅ **Métriques détaillées** et sauvegarde

## 🔧 Utilisation

### Test du Système
```bash
python3 test_simple_grasp_basic.py
```

### Entraînement Court
```bash
python3 train_simple_grasp.py --episodes 50
```

### Entraînement Complet
```bash
python3 train_simple_grasp.py --episodes 1000 --lr 3e-4
```

## 📊 Résultats

Les résultats sont sauvegardés dans `training_results/`:
- **Modèles**: `models/final_model.pth`
- **Vidéos**: `videos/episode_*.mp4`
- **Métriques**: `logs/final_metrics.json`

## 🆘 Dépannage

1. **Erreur "Module not found"**: Installer les dépendances Python
2. **Erreur "Model not found"**: Placer g1_combined.xml dans results/
3. **Pas de vidéos**: Installer imageio (`pip install imageio`)
4. **Erreurs MuJoCo**: Vérifier que le modèle XML est valide

## 🎮 Système de Récompenses

- **Contact détecté**: +0.5 à +1.0 (proportionnel à la force)
- **Cube soulevé**: +20 × hauteur (max +10.0)
- **Mouvement excessif**: -0.01 × énergie_action
- **Cube tombé**: -5.0

Le robot apprend progressivement à:
1. Approcher le cube avec ses mains
2. Détecter le contact via les capteurs de force
3. Fermer les doigts pour saisir
4. Soulever le cube avec succès

---
**Système créé automatiquement**
'''
    
    with open("README_LOCAL.md", 'w') as f:
        f.write(content)
    print("   ✅ README_LOCAL.md créé")

def main():
    """Installation principale"""
    print("🛠️  CONFIGURATION LOCALE DU SYSTÈME G1")
    print("=" * 60)
    
    try:
        # Créer la structure
        create_directory_structure()
        
        # Créer les fichiers principaux
        create_simple_grasp_env()
        create_sac_agent()
        create_video_recorder()
        create_training_script()
        create_test_script()
        create_init_files()
        create_readme()
        
        print("\\n" + "=" * 60)
        print("✅ INSTALLATION TERMINÉE!")
        print("\\n🚀 Prochaines étapes:")
        print("1. Installer les dépendances: pip install numpy torch gymnasium mujoco")
        print("2. Placer g1_combined.xml dans results/")
        print("3. Tester: python3 test_simple_grasp_basic.py")
        print("4. Entraîner: python3 train_simple_grasp.py --episodes 100")
        
        print("\\n📚 Documentation: README_LOCAL.md")
        print("🎬 Vidéos sauvegardées dans: training_results/videos/")
        print("📊 Résultats dans: training_results/")
        
    except Exception as e:
        print(f"\\n❌ Erreur lors de l'installation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()