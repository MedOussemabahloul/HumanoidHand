
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Buffer de replay pour SAC"""
  
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample un batch"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Réseau acteur pour SAC"""
  
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64], max_action=1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # Réseau principal
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        self.network = nn.Sequential(*layers)
        
        # Têtes pour moyenne et log std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass"""
        x = self.network(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        """Sample une action avec reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Appliquer tanh pour borner l'action
        action = torch.tanh(x_t)
        
        # Calculer log prob avec correction pour tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        action = action * self.max_action
        
        return action, log_prob, mean

class Critic(nn.Module):
    """Réseau critique (double Q-network)"""
  
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        super(Critic, self).__init__()
        
        # Q1 network
        layers1 = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers1.append(nn.Linear(input_dim, hidden_size))
            layers1.append(nn.ReLU())
            input_dim = hidden_size
        layers1.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        
        # Q2 network
        layers2 = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers2.append(nn.Linear(input_dim, hidden_size))
            layers2.append(nn.ReLU())
            input_dim = hidden_size
        layers2.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers2)
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
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
    """Agent SAC amélioré avec correction PyTorch"""
  
    def __init__(self, state_dim, action_dim, lr=3e-4, hidden_sizes=[64, 64],
                buffer_size=100000, gamma=0.99, tau=0.005, alpha=0.2):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Réseaux
        self.actor = Actor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        
        # Copier les poids vers le target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Automatic temperature tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.training_step = 0
    
    def select_action(self, state, evaluate=False):
        """Sélectionne une action"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans le buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=256):
        """Met à jour l'agent"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Conversion en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update alpha
        alpha_loss = self._update_alpha(states)
        
        # Update target networks
        self._update_target_networks()
        
        self.training_step += 1
        
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha": self.alpha,
            "alpha_loss": alpha_loss
        }
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Met à jour le critique"""
        with torch.no_grad():
            # Actions suivantes
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Q-values suivantes avec target network
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            # Target Q-value - CORRECTION PYTORCH ICI
            q_target = rewards + (~dones).float() * self.gamma * q_next
        
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
        """Met à jour l'acteur"""
        actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states):
        """Met à jour la température automatiquement"""
        with torch.no_grad():
            actions, log_probs, _ = self.actor.sample(states)
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def _update_target_networks(self):
        """Met à jour les réseaux target avec soft update"""
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

