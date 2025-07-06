import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("Import de agents/sac_agent.py réussi.")

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        print(f"Initialisation du MLP avec {input_dim=} {output_dim=} {hidden_sizes=}")
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print(f"Passage dans le forward du MLP avec {x.shape=}")
        return self.net(x)

class SACAgent:
    def __init__(self, obs_dim=2, act_dim=1, hidden_sizes=[64,64], lr=1e-3, alpha=0.2, gamma=0.99, tau=0.005):
        print("SACAgent : Initialisation")
        self.actor = MLP(obs_dim, act_dim, hidden_sizes)
        self.critic1 = MLP(obs_dim + act_dim, 1, hidden_sizes)
        self.critic2 = MLP(obs_dim + act_dim, 1, hidden_sizes)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

    def select_action(self, obs):
        print(f"SACAgent : select_action appelé avec obs={obs}")
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).numpy()[0]
            print(f"SACAgent : action sélectionnée = {action}")
            return action

    def update(self, replay_buffer, batch_size):
        print("SACAgent : update (dummy, non implémenté)")
