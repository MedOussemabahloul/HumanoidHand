#!/usr/bin/env python3
"""
Agent SAC amélioré pour la tâche de saisie G1
Implémentation complète avec replay buffer et entraînement
Auteur: Assistant IA
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
    
    def get_training_metrics(self):
        """Retourne les métriques d'entraînement"""
        if not self.actor_loss_history:
            return {}
        
        return {
            "actor_loss_mean": np.mean(self.actor_loss_history[-100:]),
            "critic_loss_mean": np.mean(self.critic_loss_history[-100:]),
            "alpha_mean": np.mean(self.alpha_history[-100:]),
            "training_steps": self.training_step,
            "buffer_size": len(self.replay_buffer)
        }