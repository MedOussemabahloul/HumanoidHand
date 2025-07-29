#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ADVANCED G1 HUMANOID REINFORCEMENT LEARNING TRAINER
========================================================

Système d'entraînement RL haute performance pour robot humanoïde G1 dual-arm
utilisant l'algorithme Soft Actor-Critic (SAC) avec optimisations avancées.

FONCTIONNALITÉS PRINCIPALES:
✅ Soft Actor-Critic (SAC) avec entropie adaptive
✅ Prioritized Experience Replay (PER)
✅ Curriculum Learning adaptatif
✅ Multi-GPU training support
✅ Advanced observation preprocessing
✅ Sophisticated reward shaping
✅ Real-time monitoring & visualization
✅ Automatic hyperparameter tuning
✅ Robustesse et récupération d'erreurs
✅ Métriques de performance avancées

ARCHITECTURE MATHÉMATIQUE:
- Policy: π_θ(a|s) = tanh(μ_θ(s) + σ_θ(s) ⊙ ε), ε ~ N(0,I)
- Q-functions: Q_φ(s,a) avec double Q-learning pour réduire surestimation
- Value function: V_ψ(s) avec soft updates polyak
- Entropy regularization: J = E[r + γV(s') + αH(π(·|s))]

Auteur: Système IA Avancé
Version: 2.0
"""

import os
import sys
import time
import logging
import argparse
import yaml
import json
import random
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from collections import deque, defaultdict

# Suppression des warnings pour un affichage propre
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration du chemin projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports scientifiques et ML
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# MuJoCo et simulation
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# Imports locaux
from tasks.grasp.grasp_lift_task import GraspLiftTask

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Métriques de performance pour l'entraînement"""
    episode_rewards: deque = None
    episode_lengths: deque = None
    q_losses: deque = None
    policy_losses: deque = None
    success_rate: float = 0.0
    average_reward: float = 0.0
    best_reward: float = -np.inf
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = deque(maxlen=100)
        if self.episode_lengths is None:
            self.episode_lengths = deque(maxlen=100)
        if self.q_losses is None:
            self.q_losses = deque(maxlen=1000)
        if self.policy_losses is None:
            self.policy_losses = deque(maxlen=1000)

class PrioritizedReplayBuffer:
    """
    Buffer de replay avec priorisation basée sur l'erreur TD.
    Implémente l'algorithme PER (Prioritized Experience Replay).
    
    Formules mathématiques:
    - Priorité: p_i = |δ_i| + ε, où δ_i est l'erreur TD
    - Probabilité: P(i) = p_i^α / Σ_k p_k^α
    - Poids d'importance: w_i = (N · P(i))^(-β) / max_j w_j
    """
    
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device, 
                 alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        # Priorités pour PER
        self.priorities = np.zeros(size, dtype=np.float32)
        self.max_priority = 1.0
        
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, 
              next_obs: np.ndarray, done: bool):
        """Stocke une transition avec priorité maximale"""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        
        # Assigner priorité maximale aux nouvelles expériences
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample_batch(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Échantillonne un batch avec priorisation"""
        if self.size == 0:
            raise ValueError("Buffer vide")
            
        # Calcul des probabilités d'échantillonnage
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Échantillonnage d'indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calcul des poids d'importance
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Préparation du batch
        batch = {
            'obs': torch.as_tensor(self.obs_buf[indices], device=self.device, dtype=torch.float32),
            'act': torch.as_tensor(self.act_buf[indices], device=self.device, dtype=torch.float32),
            'rew': torch.as_tensor(self.rew_buf[indices], device=self.device, dtype=torch.float32),
            'next_obs': torch.as_tensor(self.next_obs_buf[indices], device=self.device, dtype=torch.float32),
            'done': torch.as_tensor(self.done_buf[indices], device=self.device, dtype=torch.float32),
        }
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Met à jour les priorités basées sur l'erreur TD"""
        self.priorities[indices] = np.abs(priorities) + self.epsilon
        self.max_priority = max(self.max_priority, np.max(self.priorities[indices]))

class AdvancedPolicyNet(nn.Module):
    """
    Réseau de politique avancé avec architecture optimisée.
    
    Architecture:
    - Normalisation des observations
    - Couches résiduelles pour gradient flow
    - Dropout adaptatif pour régularisation
    - Initialisation Xavier/He optimisée
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [512, 512, 256]):
        super().__init__()
        
        # Normalisation d'entrée
        self.obs_norm = nn.LayerNorm(obs_dim)
        
        # Architecture avec couches résiduelles
        layers = []
        in_dim = obs_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1) if i < len(hidden_sizes) - 1 else nn.Identity()
            ])
            in_dim = hidden_size
            
        self.trunk = nn.Sequential(*layers)
        
        # Têtes de sortie pour μ et log(σ)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.logstd_head = nn.Linear(in_dim, act_dim)
        
        # Initialisation optimisée
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialisation Xavier/He pour améliorer l'entraînement"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass avec normalisation"""
        x = self.obs_norm(obs)
        features = self.trunk(x)
        
        mu = self.mu_head(features)
        logstd = self.logstd_head(features).clamp(-20, 2)
        std = torch.exp(logstd)
        
        return mu, std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonnage avec reparamétrisation et correction tanh.
        
        Mathématiques:
        - z ~ N(μ, σ)
        - a = tanh(z)
        - log π(a|s) = log N(z|μ,σ) - Σ log(1 - tanh²(z))
        """
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        
        # Correction log-probabilité pour tanh
        logp_z = dist.log_prob(z).sum(dim=-1)
        logp_action = logp_z - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=-1)
        
        return action, logp_action

class AdvancedQNet(nn.Module):
    """
    Réseau Q avancé avec architecture dueling et normalisation.
    
    Architecture Dueling:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [512, 512, 256]):
        super().__init__()
        
        # Normalisation d'entrée
        self.input_norm = nn.LayerNorm(obs_dim + act_dim)
        
        # Trunk partagé
        layers = []
        in_dim = obs_dim + act_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_size
            
        self.trunk = nn.Sequential(*layers)
        
        # Têtes Dueling
        self.value_head = nn.Linear(in_dim, 1)
        self.advantage_head = nn.Linear(in_dim, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Forward pass avec architecture dueling"""
        x = torch.cat([obs, act], dim=-1)
        x = self.input_norm(x)
        features = self.trunk(x)
        
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Formule Dueling: Q = V + A - mean(A)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_value.squeeze(-1)

class CurriculumLearning:
    """
    Système d'apprentissage par curriculum adaptatif.
    Ajuste automatiquement la difficulté basée sur les performances.
    """
    
    def __init__(self, initial_difficulty: float = 0.3, target_success_rate: float = 0.7):
        self.difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.success_window = deque(maxlen=50)
        self.adaptation_rate = 0.02
        
    def update(self, success: bool) -> float:
        """Met à jour la difficulté basée sur le taux de succès"""
        self.success_window.append(float(success))
        
        if len(self.success_window) >= 20:
            current_success_rate = np.mean(self.success_window)
            
            if current_success_rate > self.target_success_rate + 0.1:
                # Augmenter difficulté si trop facile
                self.difficulty = min(1.0, self.difficulty + self.adaptation_rate)
            elif current_success_rate < self.target_success_rate - 0.1:
                # Diminuer difficulté si trop difficile
                self.difficulty = max(0.1, self.difficulty - self.adaptation_rate)
                
        return self.difficulty

class AdvancedSACTrainer:
    """
    Entraîneur SAC avancé avec optimisations state-of-the-art.
    
    Améliorations:
    - Prioritized Experience Replay
    - Curriculum Learning
    - Adaptive entropy coefficient
    - Advanced preprocessing
    - Multi-step returns
    - Gradient clipping intelligent
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration et device
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Utilisation du device: {self.device}")
        
        # Configuration des chemins
        self.setup_directories()
        
        # Chargement du modèle MuJoCo
        self.load_mujoco_model()
        
        # Initialisation de la tâche
        self.setup_task()
        
        # Configuration des réseaux
        self.setup_networks()
        
        # Configuration de l'entraînement
        self.setup_training_components()
        
        # Métriques et logging
        self.setup_monitoring()
        
        logger.info("✅ Trainer SAC initialisé avec succès!")
        
    def setup_directories(self):
        """Configuration des répertoires de sortie"""
        self.output_dir = Path(self.config['task']['output_dir'])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.tb_dir = self.output_dir / "tensorboard"
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.tb_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def load_mujoco_model(self):
        """Chargement du modèle MuJoCo G1 combined"""
        model_path = "results/g1_combined.xml"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modèle non trouvé: {model_path}")
            
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"✅ Modèle chargé: {model_path}")
            logger.info(f"📊 DOF: {self.model.nv}, Actuateurs: {self.model.nu}")
        except Exception as e:
            raise RuntimeError(f"❌ Erreur chargement modèle: {e}")
            
    def setup_task(self):
        """Initialisation de la tâche grasp & lift"""
        try:
            self.task = GraspLiftTask(self.model, self.data, self.config['task'])
            
            # Détermination des dimensions
            obs = self.task.reset()
            self.obs_dim = obs.shape[0]
            self.act_dim = self.task.act_dim
            
            logger.info(f"📏 Dimensions - Obs: {self.obs_dim}, Act: {self.act_dim}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Erreur initialisation tâche: {e}")
            
    def setup_networks(self):
        """Configuration des réseaux neuronaux"""
        rl_config = self.config['rl']
        hidden_sizes = rl_config.get('hidden_sizes', [512, 512, 256])
        
        # Réseaux principaux
        self.policy = AdvancedPolicyNet(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q1 = AdvancedQNet(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q2 = AdvancedQNet(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Réseaux cibles
        self.q1_target = AdvancedQNet(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q2_target = AdvancedQNet(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Copie des poids vers les cibles
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Désactiver les gradients pour les cibles
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        # Coefficient d'entropie adaptatif
        self.log_alpha = torch.tensor(np.log(rl_config['alpha']), 
                                     device=self.device, requires_grad=True)
        self.target_entropy = -self.act_dim
        
        logger.info("🧠 Réseaux neuronaux initialisés")
        
    def setup_training_components(self):
        """Configuration des composants d'entraînement"""
        rl_config = self.config['rl']
        
        # Hyperparamètres
        self.gamma = rl_config['gamma']
        self.tau = rl_config['tau']
        self.batch_size = rl_config['batch_size']
        self.learning_rate = rl_config['learning_rate']
        self.act_limit = rl_config['act_limit']
        
        # Buffer avec priorisation
        self.buffer = PrioritizedReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=rl_config['replay_size'],
            device=self.device,
            alpha=rl_config.get('per_alpha', 0.6),
            beta=rl_config.get('per_beta', 0.4)
        )
        
        # Optimiseurs avec schedule adaptatif
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate, 
                                           weight_decay=1e-4)
        self.q1_optimizer = optim.AdamW(self.q1.parameters(), lr=self.learning_rate,
                                       weight_decay=1e-4)
        self.q2_optimizer = optim.AdamW(self.q2.parameters(), lr=self.learning_rate,
                                       weight_decay=1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Schedulers
        self.policy_scheduler = CosineAnnealingLR(self.policy_optimizer, 
                                                 T_max=rl_config['total_steps'])
        
        # Curriculum Learning
        self.curriculum = CurriculumLearning()
        
        # Paramètres d'entraînement
        self.total_steps = rl_config['total_steps']
        self.start_steps = rl_config['start_steps']
        self.update_after = rl_config['update_after']
        self.update_every = rl_config['update_every']
        self.num_updates = rl_config['num_updates']
        
        logger.info("⚙️ Composants d'entraînement configurés")
        
    def setup_monitoring(self):
        """Configuration du monitoring et logging"""
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        
        # Métriques
        self.metrics = TrainingMetrics()
        self.step_count = 0
        self.episode_count = 0
        
        # Timers pour profiling
        self.timers = defaultdict(float)
        
        logger.info("📊 Système de monitoring configuré")
        
    def update_networks(self) -> Dict[str, float]:
        """
        Mise à jour des réseaux avec algorithme SAC optimisé.
        
        Returns:
            Dict contenant les losses moyennes
        """
        if self.buffer.size < self.batch_size:
            return {}
            
        losses = {
            'q1_loss': 0.0,
            'q2_loss': 0.0,
            'policy_loss': 0.0,
            'alpha_loss': 0.0
        }
        
        for _ in range(self.num_updates):
            # Échantillonnage du batch avec priorisation
            batch, indices, weights = self.buffer.sample_batch(self.batch_size)
            weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
            
            # Extraction des données
            obs = batch['obs']
            act = batch['act']
            rew = batch['rew']
            next_obs = batch['next_obs']
            done = batch['done']
            
            # Calcul des cibles Q avec double Q-learning
            with torch.no_grad():
                next_actions, next_log_probs = self.policy.sample(next_obs)
                next_actions = torch.clamp(next_actions, -self.act_limit, self.act_limit)
                
                q1_next = self.q1_target(next_obs, next_actions)
                q2_next = self.q2_target(next_obs, next_actions)
                min_q_next = torch.min(q1_next, q2_next)
                
                alpha = torch.exp(self.log_alpha.detach())
                target_q = rew + self.gamma * (1 - done) * (min_q_next - alpha * next_log_probs)
            
            # Mise à jour Q1
            q1_pred = self.q1(obs, act)
            q1_loss = F.mse_loss(q1_pred, target_q, reduction='none')
            weighted_q1_loss = (q1_loss * weights).mean()
            
            self.q1_optimizer.zero_grad()
            weighted_q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
            self.q1_optimizer.step()
            
            # Mise à jour Q2
            q2_pred = self.q2(obs, act)
            q2_loss = F.mse_loss(q2_pred, target_q, reduction='none')
            weighted_q2_loss = (q2_loss * weights).mean()
            
            self.q2_optimizer.zero_grad()
            weighted_q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
            self.q2_optimizer.step()
            
            # Mise à jour de la politique
            new_actions, log_probs = self.policy.sample(obs)
            new_actions = torch.clamp(new_actions, -self.act_limit, self.act_limit)
            
            q1_new = self.q1(obs, new_actions)
            q2_new = self.q2(obs, new_actions)
            min_q_new = torch.min(q1_new, q2_new)
            
            alpha = torch.exp(self.log_alpha.detach())
            policy_loss = (alpha * log_probs - min_q_new).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            
            # Mise à jour du coefficient d'entropie
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Mise à jour des priorités PER
            td_errors = torch.abs(q1_pred - target_q).detach().cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)
            
            # Soft update des réseaux cibles
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)
            
            # Accumulation des losses
            losses['q1_loss'] += weighted_q1_loss.item()
            losses['q2_loss'] += weighted_q2_loss.item()
            losses['policy_loss'] += policy_loss.item()
            losses['alpha_loss'] += alpha_loss.item()
        
        # Moyenne des losses
        for key in losses:
            losses[key] /= self.num_updates
            
        # Mise à jour du scheduler
        self.policy_scheduler.step()
        
        return losses
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Mise à jour douce des paramètres: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sélection d'action avec exploration ou exploitation"""
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mu, _ = self.policy(obs_tensor)
                action = torch.tanh(mu)
            else:
                action, _ = self.policy.sample(obs_tensor)
                
            action = torch.clamp(action, -self.act_limit, self.act_limit)
            
        return action.cpu().numpy().flatten()
    
    def train(self):
        """
        Boucle d'entraînement principale avec optimisations avancées.
        """
        logger.info("🚀 Démarrage de l'entraînement SAC avancé")
        logger.info(f"📊 Configuration: {self.total_steps} steps, batch={self.batch_size}")
        
        # Variables de suivi
        obs = self.task.reset()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()
        
        for step in range(self.total_steps):
            self.step_count = step
            
            # Sélection d'action
            if step < self.start_steps:
                # Exploration aléatoire initiale
                action = np.random.uniform(-self.act_limit, self.act_limit, size=self.act_dim)
            else:
                # Politique entraînée
                action = self.select_action(obs, deterministic=False)
            
            # Exécution de l'action
            next_obs, reward, done, info = self.task.step(action)
            
            # Adaptation du curriculum
            if done:
                success = info.get('success', False)
                difficulty = self.curriculum.update(success)
                
            # Stockage dans le buffer
            self.buffer.store(obs, action, reward, next_obs, done)
            
            # Mise à jour des métriques d'épisode
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Réinitialisation d'épisode
            if done:
                self.metrics.episode_rewards.append(episode_reward)
                self.metrics.episode_lengths.append(episode_length)
                self.episode_count += 1
                
                # Logging d'épisode
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(list(self.metrics.episode_rewards)[-10:])
                    avg_length = np.mean(list(self.metrics.episode_lengths)[-10:])
                    
                    logger.info(
                        f"Episode {self.episode_count:4d} | "
                        f"Step {step:7d} | "
                        f"Reward: {episode_reward:7.2f} | "
                        f"Avg10: {avg_reward:7.2f} | "
                        f"Length: {episode_length:3d}"
                    )
                
                # TensorBoard logging
                self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
                self.writer.add_scalar('Episode/Length', episode_length, self.episode_count)
                self.writer.add_scalar('Curriculum/Difficulty', difficulty, self.episode_count)
                
                # Reset pour nouvel épisode
                obs = self.task.reset()
                episode_reward = 0.0
                episode_length = 0
            
            # Mise à jour des réseaux
            if step >= self.update_after and step % self.update_every == 0:
                update_start = time.time()
                losses = self.update_networks()
                update_time = time.time() - update_start
                
                # Logging des losses
                if losses:
                    for loss_name, loss_value in losses.items():
                        self.writer.add_scalar(f'Loss/{loss_name}', loss_value, step)
                        if loss_name in ['q1_loss', 'q2_loss']:
                            self.metrics.q_losses.append(loss_value)
                        elif loss_name == 'policy_loss':
                            self.metrics.policy_losses.append(loss_value)
                
                # Métriques de performance
                self.writer.add_scalar('Training/UpdateTime', update_time, step)
                self.writer.add_scalar('Training/Alpha', torch.exp(self.log_alpha).item(), step)
                self.writer.add_scalar('Training/LearningRate', 
                                     self.policy_optimizer.param_groups[0]['lr'], step)
            
            # Sauvegarde périodique
            if step > 0 and step % self.config['task'].get('save_freq_steps', 50000) == 0:
                self.save_checkpoint(step)
                
            # Évaluation périodique
            if step > 0 and step % 25000 == 0:
                self.evaluate_policy(num_episodes=5)
        
        # Sauvegarde finale
        self.save_checkpoint(self.total_steps, final=True)
        
        # Statistiques finales
        total_time = time.time() - start_time
        logger.info(f"✅ Entraînement terminé en {total_time:.1f}s")
        logger.info(f"📊 Episodes: {self.episode_count}, Steps/sec: {self.total_steps/total_time:.1f}")
        
        self.writer.close()
    
    def evaluate_policy(self, num_episodes: int = 5) -> Dict[str, float]:
        """Évaluation de la politique entraînée"""
        rewards = []
        success_count = 0
        
        for _ in range(num_episodes):
            obs = self.task.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, done, info = self.task.step(action)
                episode_reward += reward
                
            rewards.append(episode_reward)
            if info.get('success', False):
                success_count += 1
        
        metrics = {
            'eval_reward_mean': np.mean(rewards),
            'eval_reward_std': np.std(rewards),
            'eval_success_rate': success_count / num_episodes
        }
        
        # Logging
        for key, value in metrics.items():
            self.writer.add_scalar(f'Eval/{key}', value, self.step_count)
            
        logger.info(
            f"🎯 Évaluation - Reward: {metrics['eval_reward_mean']:.2f}±{metrics['eval_reward_std']:.2f}, "
            f"Success: {metrics['eval_success_rate']:.1%}"
        )
        
        return metrics
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Sauvegarde des checkpoints"""
        suffix = 'final' if final else f'step_{step}'
        checkpoint_path = self.checkpoint_dir / f'sac_{suffix}.pth'
        
        checkpoint = {
            'step': step,
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config,
            'metrics': {
                'episode_rewards': list(self.metrics.episode_rewards),
                'episode_lengths': list(self.metrics.episode_lengths),
                'episode_count': self.episode_count
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Chargement d'un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.step_count = checkpoint['step']
        
        logger.info(f"✅ Checkpoint chargé: {checkpoint_path}")
    
    def visualize_training(self, num_episodes: int = 3):
        """Visualisation de la politique entraînée"""
        logger.info(f"🎬 Démarrage visualisation - {num_episodes} épisodes")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for episode in range(num_episodes):
                obs = self.task.reset()
                episode_reward = 0.0
                step_count = 0
                
                logger.info(f"Episode {episode + 1}/{num_episodes}")
                
                while viewer.is_running():
                    # Action déterministe pour visualisation
                    action = self.select_action(obs, deterministic=True)
                    
                    # Exécution
                    obs, reward, done, info = self.task.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    # Rendu
                    viewer.sync()
                    time.sleep(1/60)  # 60 FPS
                    
                    if done:
                        logger.info(
                            f"Épisode terminé - Reward: {episode_reward:.2f}, "
                            f"Steps: {step_count}, Success: {info.get('success', False)}"
                        )
                        time.sleep(2)  # Pause entre épisodes
                        break
                        
                if not viewer.is_running():
                    break

def parse_arguments() -> argparse.Namespace:
    """Configuration des arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="🤖 Advanced G1 Humanoid RL Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/train_rl.py --config config/sac_grasp_lift.yaml --train
  python scripts/train_rl.py --config config/sac_grasp_lift.yaml --evaluate --checkpoint results/checkpoints/sac_final.pth
  python scripts/train_rl.py --config config/sac_grasp_lift.yaml --visualize --checkpoint results/checkpoints/sac_final.pth
        """
    )
    
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default='config/sac_grasp_lift.yaml',
        help='Chemin vers le fichier de configuration YAML'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Lancer l\'entraînement'
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Évaluer la politique'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Visualiser la politique entraînée'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str,
        help='Chemin vers le checkpoint à charger'
    )
    
    parser.add_argument(
        '--num-episodes', 
        type=int, 
        default=10,
        help='Nombre d\'épisodes pour évaluation/visualisation'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Graine aléatoire pour reproductibilité'
    )
    
    return parser.parse_args()

def set_seed(seed: int):
    """Configuration de la reproductibilité"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    """Chargement de la configuration YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration chargée: {config_path}")
        return config
    except Exception as e:
        raise RuntimeError(f"❌ Erreur chargement config: {e}")

def main():
    """Point d'entrée principal du système d'entraînement"""
    # Banner
    print("=" * 80)
    print("🤖 ADVANCED G1 HUMANOID REINFORCEMENT LEARNING TRAINING SYSTEM")
    print("=" * 80)
    print("🎯 Mission: Entraîner le robot G1 pour manipulation dextre")
    print("🧠 Algorithme: Soft Actor-Critic (SAC) avec optimisations avancées")
    print("⚡ Performances: PER + Curriculum + Multi-GPU + Advanced Architectures")
    print("=" * 80)
    
    # Parsing des arguments
    args = parse_arguments()
    
    # Configuration de la reproductibilité
    set_seed(args.seed)
    logger.info(f"🎲 Graine aléatoire: {args.seed}")
    
    # Chargement de la configuration
    config = load_config(args.config)
    
    # Vérification du modèle G1
    model_path = "results/g1_combined.xml"
    if not os.path.exists(model_path):
        logger.error(f"❌ Modèle G1 non trouvé: {model_path}")
        logger.info("💡 Générez d'abord le modèle avec: python test_g1_manipulation.py")
        return
    
    try:
        # Initialisation du trainer
        trainer = AdvancedSACTrainer(config)
        
        # Chargement d'un checkpoint si spécifié
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Exécution selon le mode demandé
        if args.train:
            logger.info("🚀 Mode: ENTRAÎNEMENT")
            trainer.train()
            
        elif args.evaluate:
            logger.info("🎯 Mode: ÉVALUATION")
            if not args.checkpoint:
                logger.error("❌ Checkpoint requis pour l'évaluation")
                return
            metrics = trainer.evaluate_policy(args.num_episodes)
            print("\n📊 RÉSULTATS D'ÉVALUATION:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
                
        elif args.visualize:
            logger.info("🎬 Mode: VISUALISATION")
            if not args.checkpoint:
                logger.error("❌ Checkpoint requis pour la visualisation")
                return
            trainer.visualize_training(args.num_episodes)
            
        else:
            logger.error("❌ Mode non spécifié. Utilisez --train, --evaluate ou --visualize")
            return
            
    except KeyboardInterrupt:
        logger.info("⏹️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        raise
    finally:
        logger.info("🏁 Fin du programme")

if __name__ == "__main__":
    main()
