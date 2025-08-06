#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 TRAIN CORRECTED ULTRA STABLE - ENTRAÎNEMENT SAC AVEC SYSTÈME CORRIGÉ
=======================================================================

Entraînement SAC ultra-stable avec le système G1 corrigé qui bloque les DOFs
problématiques des doigts et ne contrôle que les DOFs des bras.

AMÉLIORATIONS:
✅ Système corrigé: Doigts bloqués, bras contrôlables
✅ SAC ultra-stable: Entropie adaptative, double Q-learning
✅ Monitoring avancé: Métriques temps réel, logging structuré
✅ Gestion erreurs: Recovery automatique, validation continue
✅ Performance optimisée: Actions limitées, observations filtrées

ARCHITECTURE SAC:
- Actor: π_θ(a|s) = tanh(μ_θ(s) + σ_θ(s) ⊙ ε), ε ~ N(0,I)
- Critics: Q_φ₁(s,a), Q_φ₂(s,a) avec double Q-learning
- Entropy: α adaptative via J_α = -α(log π + H_target)

Version: 1.0 - Ultra Stable
"""

import os
import sys
import time
import logging
import argparse
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from collections import deque
import json

# Suppression warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration projet
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Imports scientifiques
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Import système corrigé
from test_corrected_final import G1CorrectedSystem

# Configuration logging
def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configuration logging avancé"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('SAC_CORRECTED')
    logger.setLevel(level)
    
    # Handler fichier
    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setLevel(level)
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class SACNetwork(nn.Module):
    """Réseau de base pour SAC"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        return self.network(x)

class SACCritic(nn.Module):
    """Critique SAC avec double Q-learning"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.q1 = nn.Sequential(
            SACNetwork(obs_dim + action_dim, hidden_dims),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.q2 = nn.Sequential(
            SACNetwork(obs_dim + action_dim, hidden_dims),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

class SACPolicyNetwork(nn.Module):
    """Réseau de politique SAC avec entropie"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.backbone = SACNetwork(obs_dim, hidden_dims)
        self.mean_layer = nn.Linear(self.backbone.output_dim, action_dim)
        self.log_std_layer = nn.Linear(self.backbone.output_dim, action_dim)
        
        self.action_scale = 0.03  # Limite d'action du système corrigé
        self.reparam_noise = 1e-6
        
    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        
        # Log probability avec correction Jacobien
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale
        
        return action, log_prob, mean

class SACAgent:
    """Agent SAC ultra-stable pour système corrigé"""
    
    def __init__(self, obs_dim: int, action_dim: int, device: str = 'cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Réseaux
        self.policy = SACPolicyNetwork(obs_dim, action_dim).to(device)
        self.critic = SACCritic(obs_dim, action_dim).to(device)
        self.target_critic = SACCritic(obs_dim, action_dim).to(device)
        
        # Copie des poids vers target
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimiseurs
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Entropie adaptative
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Hyperparamètres
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        
        # Buffer de replay simple
        self.replay_buffer = deque(maxlen=100000)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, deterministic=False):
        """Sélectionne une action"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                _, _, action = self.policy.sample(obs_tensor)
            else:
                action, _, _ = self.policy.sample(obs_tensor)
                
        return action.cpu().numpy()[0]
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Stocke une transition"""
        self.replay_buffer.append((obs, action, reward, next_obs, done))
    
    def update(self):
        """Met à jour l'agent SAC"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Échantillonnage du batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        transitions = [self.replay_buffer[i] for i in batch]
        
        obs = torch.FloatTensor([t[0] for t in transitions]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in transitions]).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor([t[3] for t in transitions]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in transitions]).unsqueeze(1).to(self.device)
        
        # Update critique
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_obs)
            target_q1, target_q2 = self.target_critic(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update politique
        new_actions, log_probs, _ = self.policy.sample(obs)
        q1, q2 = self.critic(obs, new_actions)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update alpha (entropie)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_mean': q.mean().item()
        }

class TrainingManager:
    """Gestionnaire d'entraînement ultra-stable"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Logging
        self.log_dir = Path(config['log_dir'])
        self.logger = setup_logging(self.log_dir)
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Environnement corrigé
        try:
            self.env = G1CorrectedSystem()
            self.logger.info(f"✅ Environnement corrigé initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation environnement: {e}")
            raise
        
        # Agent SAC
        self.agent = SACAgent(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            device=self.device
        )
        
        # Métriques
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.instabilities = deque(maxlen=100)
        
        self.logger.info(f"🚀 Agent SAC initialisé: obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")
        
    def train_episode(self, episode: int) -> Dict[str, float]:
        """Entraîne un épisode"""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_instabilities = 0
        
        max_steps = self.config.get('max_steps', 20)
        
        for step in range(max_steps):
            # Sélection d'action
            action = self.agent.select_action(obs, deterministic=False)
            
            # Step environnement
            try:
                next_obs, reward, done, info = self.env.step(action)
                
                # Stockage transition
                self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Mise à jour métriques
                episode_reward += reward
                episode_length += 1
                episode_instabilities += info.get('instabilities', 0)
                
                obs = next_obs
                
                # Terminaison
                if done:
                    break
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur step {step}: {e}")
                break
        
        # Update agent
        update_info = self.agent.update()
        
        # Stockage métriques
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.instabilities.append(episode_instabilities)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'instabilities': episode_instabilities,
            **update_info
        }
    
    def train(self):
        """Entraînement principal"""
        episodes = self.config.get('episodes', 50)
        
        self.logger.info(f"🚀 Début entraînement: {episodes} épisodes")
        
        for episode in range(episodes):
            start_time = time.time()
            
            try:
                # Entraînement épisode
                metrics = self.train_episode(episode)
                
                # Logging
                if episode % 5 == 0 or episode < 10:
                    avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                    avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                    avg_instabilities = np.mean(self.instabilities) if self.instabilities else 0
                    
                    self.logger.info(
                        f"Episode {episode:3d}: "
                        f"R={metrics['reward']:6.2f} "
                        f"L={metrics['length']:2d} "
                        f"I={metrics['instabilities']:2d} "
                        f"AvgR={avg_reward:6.2f} "
                        f"AvgL={avg_length:4.1f} "
                        f"AvgI={avg_instabilities:4.1f}"
                    )
                
                # TensorBoard
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'train/{key}', value, episode)
                
                # Validation périodique
                if episode % 20 == 0 and episode > 0:
                    self._validate_system(episode)
                
            except Exception as e:
                self.logger.error(f"❌ Erreur épisode {episode}: {e}")
                traceback.print_exc()
                continue
        
        self.logger.info("✅ Entraînement terminé")
        self._save_results()
    
    def _validate_system(self, episode: int):
        """Validation du système"""
        try:
            # Test de validation
            obs = self.env.reset()
            total_reward = 0
            total_instabilities = 0
            
            for _ in range(10):
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                total_instabilities += info.get('instabilities', 0)
                
                if done:
                    break
            
            self.logger.info(
                f"🔍 Validation {episode}: "
                f"Reward={total_reward:.2f}, "
                f"Instabilities={total_instabilities}"
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️  Erreur validation: {e}")
    
    def _save_results(self):
        """Sauvegarde les résultats"""
        results = {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'instabilities': list(self.instabilities),
            'config': self.config
        }
        
        results_path = self.log_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"💾 Résultats sauvegardés: {results_path}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Train SAC with Corrected G1 System')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=20, help='Max steps per episode')
    parser.add_argument('--log-dir', type=str, default='logs/corrected_training', help='Log directory')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'log_dir': args.log_dir
    }
    
    print("🚀 TRAIN CORRECTED ULTRA STABLE")
    print("=" * 50)
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Entraînement
        trainer = TrainingManager(config)
        trainer.train()
        
        print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())