#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agents/simple_sac_agent.py

Agent SAC (Soft Actor-Critic) simplifié pour l'apprentissage du grasping.
Implémentation basique mais fonctionnelle avec replay buffer et mise à jour des réseaux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class MLP(nn.Module):
    """
    Réseau de neurones multi-couches simple
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Couches cachées
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, action=None):
        if action is not None:
            # Pour les critiques qui prennent obs + action
            x = torch.cat([x, action], dim=-1)
        return self.net(x)

class ReplayBuffer:
    """
    Buffer de replay pour stocker les expériences
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience au buffer
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Échantillonne un batch d'expériences
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class SimpleSACAgent:
    """
    Agent SAC simplifié pour l'apprentissage du grasping
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256], lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):
        """
        Initialise l'agent SAC
        
        Args:
            obs_dim: Dimension de l'observation
            act_dim: Dimension de l'action
            hidden_sizes: Tailles des couches cachées
            lr: Taux d'apprentissage
            gamma: Facteur de discount
            tau: Paramètre de mise à jour des réseaux cibles
            alpha: Paramètre d'entropie
            device: Device (CPU/GPU)
        """
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Réseaux de neurones
        self.actor = MLP(obs_dim, act_dim * 2, hidden_sizes).to(device)  # *2 pour mean et log_std
        self.critic1 = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.critic2 = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.critic1_target = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
        self.critic2_target = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
        
        # Copier les poids initiaux des réseaux cibles
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Buffer de replay
        self.replay_buffer = ReplayBuffer()
        
        # Compteurs
        self.total_steps = 0
        self.update_count = 0
    
    def select_action(self, obs, evaluate=False):
        """
        Sélectionne une action basée sur l'observation
        
        Args:
            obs: Observation actuelle
            evaluate: Si True, utilise la politique déterministe
            
        Returns:
            np.ndarray: Action sélectionnée
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Mode évaluation : action déterministe
                action_mean, _ = self.actor(obs).chunk(2, dim=-1)
                action = torch.tanh(action_mean)
            else:
                # Mode entraînement : action stochastique
                action_mean, action_log_std = self.actor(obs).chunk(2, dim=-1)
                action_log_std = torch.clamp(action_log_std, -20, 2)
                action_std = torch.exp(action_log_std)
                
                # Échantillonner l'action
                noise = torch.randn_like(action_mean)
                action = torch.tanh(action_mean + action_std * noise)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch_size=256):
        """
        Met à jour les réseaux de l'agent
        
        Args:
            batch_size: Taille du batch pour l'entraînement
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Échantillonner du buffer de replay
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
        
        # Convertir en tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Mise à jour des critiques
        with torch.no_grad():
            # Actions et log-probabilités pour l'état suivant
            next_action_mean, next_action_log_std = self.actor(next_state_batch).chunk(2, dim=-1)
            next_action_log_std = torch.clamp(next_action_log_std, -20, 2)
            next_action_std = torch.exp(next_action_log_std)
            
            # Échantillonner les actions suivantes
            noise = torch.randn_like(next_action_mean)
            next_action = torch.tanh(next_action_mean + next_action_std * noise)
            
            # Log-probabilité de l'action
            log_prob_next = self._log_prob(next_action, next_action_mean, next_action_log_std)
            
            # Valeurs Q cibles
            target_Q1 = self.critic1_target(next_state_batch, next_action)
            target_Q2 = self.critic2_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * log_prob_next
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q
        
        # Mise à jour du premier critique
        current_Q1 = self.critic1(state_batch, action_batch)
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Mise à jour du deuxième critique
        current_Q2 = self.critic2(state_batch, action_batch)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Mise à jour de l'acteur
        action_mean, action_log_std = self.actor(state_batch).chunk(2, dim=-1)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        action_std = torch.exp(action_log_std)
        
        # Échantillonner les actions
        noise = torch.randn_like(action_mean)
        action = torch.tanh(action_mean + action_std * noise)
        
        # Log-probabilité
        log_prob = self._log_prob(action, action_mean, action_log_std)
        
        # Valeurs Q
        Q1 = self.critic1(state_batch, action)
        Q2 = self.critic2(state_batch, action)
        Q = torch.min(Q1, Q2)
        
        # Loss de l'acteur
        actor_loss = (self.alpha * log_prob - Q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Mise à jour des réseaux cibles
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        
        self.update_count += 1
    
    def _log_prob(self, action, mean, log_std):
        """
        Calcule la log-probabilité d'une action
        """
        # Log-probabilité de la distribution normale
        log_prob = -0.5 * ((action - mean) / torch.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        
        # Correction pour la transformation tanh
        log_prob = log_prob - torch.log(1 - action ** 2 + 1e-6)
        
        return log_prob.sum(-1, keepdim=True)
    
    def _soft_update(self, target, source):
        """
        Mise à jour douce des réseaux cibles
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        """
        Sauvegarde les modèles
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath):
        """
        Charge les modèles
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
    
    def get_stats(self):
        """
        Retourne les statistiques de l'agent
        """
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'buffer_size': len(self.replay_buffer)
        }