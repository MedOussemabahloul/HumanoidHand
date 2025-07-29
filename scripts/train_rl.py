#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rl_sac.py

Entraînement Soft Actor-Critic (SAC) pour la tâche Grasp & Lift du robot G1 dual-arm,
sans gym et sans mujoco_py, en utilisant les bindings officiels mujoco et PyTorch.

Explications détaillées en commentaires à chaque étape :
  - Chargement de la config YAML (task + rl)
  - Concaténation des MJCF body + fingers → g1_combined.xml
  - Chargement de MuJoCo (MjModel, MjData)
  - Initialisation de GraspLiftTask (obs_dim, action_dim, step, reset)
  - Réseaux : Policy (Gaussian+Tanh), Q1, Q2, V et target V
  - Replay buffer circulaire
  - Boucle de collecte & mises à jour SAC
  - Formules mathématiques et justification de chaque loss
  - Logging TensorBoard et checkpoints PyTorch
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse                          # Parsing des arguments CLI
import yaml                              # Lecture du fichier de configuration YAML
import numpy as np                       # Calcul scientifique (tableaux)
import torch                             # Framework deep learning
import torch.nn as nn                    # Modules de réseaux de neurones
import torch.optim as optim              # Optimiseurs (Adam...)
import mujoco                            # Bindings officiels MuJoCo
from torch.utils.tensorboard import SummaryWriter
import xml.etree.ElementTree as ET
from copy import deepcopy
import copy
import mujoco.viewer
from tasks.grasp.grasp_lift_task import GraspLiftTask
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1) PARSER DES ARGUMENTS CLI
# -----------------------------------------------------------------------------
def parse_args():
    """
    Définit et parse les arguments CLI :
      - config     : chemin vers le YAML (parties 'task' et 'rl')
      - body_xml   : MJCF describing body (bras, base, etc.)
      - fingers_xml: MJCF contenant doigts + capteurs
      - output_dir : dossier pour logs et checkpoints
    """
    parser = argparse.ArgumentParser(
        description="Entraînement SAC sur G1 Grasp & Lift"
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to YAML config file (task + rl sections)"
    )
    parser.add_argument(
        "--body_xml", required=True,
        help="Path to G1 body MJCF (e.g., g1_body.xml)"
    )
    parser.add_argument(
        "--fingers_xml", required=True,
        help="Path to G1 fingers MJCF (e.g., g1_fingers.xml)"
    )
    parser.add_argument(
        "-o", "--output_dir", default="results",
        help="Directory for logs, model and checkpoints"
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2) CHARGEMENT DE LA CONFIGURATION YAML
# -----------------------------------------------------------------------------
def load_config(path):
    """
    Lit un fichier YAML et renvoie un dictionnaire Python.
    Le YAML doit contenir :
      task: paramètres spécifiques à la tâche (capteurs, rewards, etc.)
      rl  : hyperparamètres SAC (gamma, alpha, lr, buffer, etc.)
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# 3) CONSTRUCTION DU MJCF COMBINÉ
def build_combined_xml(body_xml: str, fingers_xml: str, out_dir: str) -> str:
    """
    Génère un MJCF unique qui :
      - inclut entièrement body_xml et fingers_xml (mesh, sites, sensors…)
      - ajoute un bloc <actuator> automatique pour chaque joint trouvé
    """
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, "g1_combined.xml")

    abs_body   = os.path.abspath(body_xml)
    abs_finger = os.path.abspath(fingers_xml)

    # 1) Parser les deux XML pour extraire la liste de joints
    tree_b = ET.parse(abs_body)
    tree_f = ET.parse(abs_finger)
    joints = {j.attrib["name"] for j in tree_b.findall(".//joint")}
    joints.update({j.attrib["name"] for j in tree_f.findall(".//joint")})

    # 2) Construire le fichier combiné uniquement avec des <include> + <actuator>
    with open(combined_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<mujoco model="g1_combined">\n')

        # inclure body + fingers (assets, worldbody, sensors intactes)
        f.write(f'  <include file="{abs_body}"/>\n')
        f.write(f'  <include file="{abs_finger}"/>\n\n')

        # générer automatiquement les actuators
        f.write('  <actuator>\n')
        for jn in sorted(joints):
            f.write(
                f'    <position name="act_{jn}" '
                f'joint="{jn}" gear="1"/>\n'
            )
        f.write('  </actuator>\n')

        f.write('</mujoco>\n')

    print(f"[INFO] Combined MJCF written to {combined_path}")
    return combined_path


# -----------------------------------------------------------------------------
# 4) DÉFINITION DES RÉSEAUX POUR SAC
# -----------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """
    Réseau d’acteur déterminant une distribution gaussienne puis tanh :
      - mu(s), sigma(s) → Normal(mu, sigma) → z = mu + sigma * eps
      - a = tanh(z) pour contraindre l’action entre -1 et 1
    Log-probabilité corrigée du tanh :
      log π(a|s) = Normal.log_prob(z) - log(1 - tanh(z)^2 + ε)
    """
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        # trunk partagé
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        # mu et log_sigma output
        self.mu_layer    = nn.Linear(hidden_size, act_dim)
        self.logstd_layer = nn.Linear(hidden_size, act_dim)

    def forward(self, obs):
        """
        Args:
          obs: Tensor [batch, obs_dim]
        Returns:
          mu  : [batch, act_dim]
          std : [batch, act_dim] = exp(logstd).clamped
        """
        h = self.fc(obs)
        mu = self.mu_layer(h)
        logstd = self.logstd_layer(h).clamp(-20, 2)  # éviter σ trop petit/grand
        std = torch.exp(logstd)
        return mu, std

    def sample(self, obs):
        """
        Generate action with reparameterization:
          z ~ Normal(mu, std)
          a = tanh(z)
          logp = log Normal(z) - sum(log(1 - tanh(z)^2 + ε))
        """
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()  # z = mu + std * ε (ε ~ N(0,1))
        action = torch.tanh(z)
        # correction jacobienne du tanh (pour la log-prob)
        logp_z = dist.log_prob(z).sum(dim=-1)
        logp_a = logp_z - (2*(np.log(2) - z - nn.functional.softplus(-2*z))).sum(dim=-1)
        return action, logp_a

class QNet(nn.Module):
    """
    Critique Q(s,a) => estime la valeur d’état-action
    Un réseau feedforward simple qui concatène obs et act.
    """
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),      nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)  # [batch,]

class ValueNet(nn.Module):
    """
    Critique V(s) => estime la valeur d’état
    """
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)  # [batch,]

# -----------------------------------------------------------------------------
# 5) REPLAY BUFFER CIRCULAIRE
# -----------------------------------------------------------------------------
class ReplayBuffer:
    """
    Stocke transitions (s,a,r,s',done) en anneau pour amortir la corrélation
    """
    def __init__(self, obs_dim, act_dim, size,device):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size,       dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size,       dtype=np.float32)
        self.max_size = size
        self.ptr      = 0
        self.size     = 0
        self.device  = device

    def store(self, obs, act, rew, nxt_obs, done):
        """
        Insère une transition et écrase l’ancienne si plein.
        """
        self.obs_buf[self.ptr]      = obs
        self.act_buf[self.ptr]      = act
        self.rew_buf[self.ptr]      = rew
        self.next_obs_buf[self.ptr] = nxt_obs
        self.done_buf[self.ptr]     = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """
        Échantillonne un batch aléatoire pour l’update SAC
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs      = torch.as_tensor(self.obs_buf[idxs],device=self.device, dtype=torch.float32),
            act      = torch.as_tensor(self.act_buf[idxs],device=self.device, dtype=torch.float32),
            rew      = torch.as_tensor(self.rew_buf[idxs],device=self.device, dtype=torch.float32),
            next_obs = torch.as_tensor(self.next_obs_buf[idxs],device=self.device, dtype=torch.float32),
            done     = torch.as_tensor(self.done_buf[idxs],device=self.device, dtype=torch.float32),
        )
        return batch

# -----------------------------------------------------------------------------
# 6) SAC TRAINER: collecte & update
# -----------------------------------------------------------------------------
class SACTrainer:
    def __init__(self, cfg, model_xml):
        # 1) Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) Config task vs SAC
        self.task_cfg   = cfg["task"]
        rl_cfg          = cfg["rl"]
        self.act_limit  = rl_cfg.get("act_limit", 1.0)

        # 3) Hyperparamètres SAC
        self.gamma        = float(rl_cfg["gamma"])
        self.alpha        = float(rl_cfg["alpha"])
        self.lr           = float(rl_cfg["learning_rate"])
        self.hidden       = int(rl_cfg["hidden_size"])
        self.batch_size   = int(rl_cfg["batch_size"])
        self.replay_size  = int(rl_cfg["replay_size"])
        self.start_steps  = int(rl_cfg["start_steps"])
        self.update_after = int(rl_cfg["update_after"])
        self.update_every = int(rl_cfg["update_every"])
        self.num_updates  = int(rl_cfg["num_updates"])
        self.total_steps  = int(rl_cfg["total_steps"])
        self.tau          = float(rl_cfg.get("tau", 0.005))

        # 4) Charger MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data  = mujoco.MjData(self.model)

        # 5) GraspLiftTask
        self.task = GraspLiftTask(self.model, self.data, self.task_cfg)

        # 6) Dimensions dynamiques
        obs           = self.task.reset()
        self.obs_dim  = obs.shape[0]
        self.act_dim  = self.task.act_dim

        # 7) ReplayBuffer
        self.buffer = ReplayBuffer(
            obs_dim = self.obs_dim,
            act_dim = self.act_dim,
            size    = self.replay_size,
            device  = self.device
        )

        # 8) Réseaux principaux
        self.policy = PolicyNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
        self.q1      = QNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
        self.q2      = QNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
        self.v       = ValueNet(self.obs_dim, self.hidden).to(self.device)

        # 9) Cibles pour V
        self.v_targ = ValueNet(self.obs_dim, self.hidden).to(self.device)
        self.v_targ.load_state_dict(self.v.state_dict())
        for p in self.v_targ.parameters():
            p.requires_grad = False

        # 10) Cibles pour Q
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # 11) Optimizers
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(),     lr=self.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(),     lr=self.lr)
        self.v_opt  = optim.Adam(self.v.parameters(),      lr=self.lr)

        # 12) Logger TensorBoard
        tb_dir = os.path.join(self.task_cfg.get("output_dir", "results"), "tb")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer     = SummaryWriter(log_dir=tb_dir)
        self.step_count = 0

    def update(self):
        # 1) Pré-initialisation des losses pour être sûrs qu’elles existent
        total_q1_loss = torch.tensor(0.0, device=self.device)
        total_q2_loss = torch.tensor(0.0, device=self.device)
        total_pi_loss = torch.tensor(0.0, device=self.device)

        for i in range(self.num_updates):
            batch = self.buffer.sample_batch(self.batch_size)
            obs, act = batch["obs"], batch["act"]
            rew, nxt, done = batch["rew"], batch["next_obs"], batch["done"]

            # to(device)
            obs  = torch.as_tensor(obs,  device=self.device, dtype=torch.float32)
            act  = torch.as_tensor(act,  device=self.device, dtype=torch.float32)
            rew  = torch.as_tensor(rew,  device=self.device, dtype=torch.float32).unsqueeze(-1)
            nxt  = torch.as_tensor(nxt,  device=self.device, dtype=torch.float32)
            done = torch.as_tensor(done, device=self.device, dtype=torch.float32).unsqueeze(-1)

            # ----- 1) Critic update -----
            with torch.no_grad():
                a2, logp2 = self.policy.sample(nxt)
                a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

                q1_pi_t = self.q1_target(nxt, a2)
                q2_pi_t = self.q2_target(nxt, a2)
                min_q_t = torch.min(q1_pi_t, q2_pi_t)

                # target_val [batch,1]
                target_val = (min_q_t - self.alpha * logp2).unsqueeze(-1)
                q_backup   = rew + self.gamma * (1 - done) * target_val

            # ----- Q1 loss -----
            q1_pred = self.q1(obs, act)
            if q1_pred.dim() == 1:
                q1_pred = q1_pred.unsqueeze(-1)
            q1_loss = F.mse_loss(q1_pred, q_backup)
            self.q1_opt.zero_grad()
            q1_loss.backward()
            self.q1_opt.step()
            total_q1_loss += q1_loss.detach()

            # ----- Q2 loss -----
            q2_pred = self.q2(obs, act)
            if q2_pred.dim() == 1:
                q2_pred = q2_pred.unsqueeze(-1)
            q2_loss = F.mse_loss(q2_pred, q_backup)
            self.q2_opt.zero_grad()
            q2_loss.backward()
            self.q2_opt.step()
            total_q2_loss += q2_loss.detach()

            # ----- 2) Policy update -----
            a_new, logp = self.policy.sample(obs)
            a_new = torch.clamp(a_new, -self.act_limit, self.act_limit)

            q1_pi    = self.q1(obs, a_new).detach()
            q2_pi    = self.q2(obs, a_new).detach()
            min_q_pi = torch.min(q1_pi, q2_pi)

            pi_loss = (self.alpha * logp - min_q_pi).mean()
            self.pi_opt.zero_grad()
            pi_loss.backward()
            self.pi_opt.step()
            total_pi_loss += pi_loss.detach()

            # ----- 3) Soft-updates des cibles -----
            with torch.no_grad():
                for p, p_targ in zip(self.v.parameters(),   self.v_targ.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_( self.tau * p.data)
                for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_( self.tau * p.data)
                for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_( self.tau * p.data)

        # 4) Logging : on divise par num_updates pour avoir la moyenne
        avg_q1 = (total_q1_loss / self.num_updates).item()
        avg_q2 = (total_q2_loss / self.num_updates).item()
        avg_pi = (total_pi_loss / self.num_updates).item()

        self.writer.add_scalar("loss/q1_loss", avg_q1, self.step_count)
        self.writer.add_scalar("loss/q2_loss", avg_q2, self.step_count)
        self.writer.add_scalar("loss/pi_loss", avg_pi, self.step_count)



    def train(self):
        
        obs = self.task.reset()
        for t in range(self.total_steps):
            self.step_count = t

            # 1) Exploration vs exploitation
            if t < self.start_steps:
                action = np.random.uniform(-1, 1, size=self.act_dim) * self.act_limit
            else:
                with torch.no_grad():
                    a_t, _ = self.policy.sample(
                        torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    )
                    action = a_t.cpu().numpy()
            # clamp avant simuler
            action = np.clip(action, -self.act_limit, self.act_limit)

            # 2) Simuler
            next_obs, reward, done, _ = self.task.step(action)

            # 3) Stocker
            self.buffer.store(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else self.task.reset()

            # 4) Update périodique
            if t >= self.update_after and t % self.update_every == 0:
                self.update()

            # 5) Log environnement
            self.writer.add_scalar("env/reward", reward, t)

            # 6) Checkpoint
            if t > 0 and t % self.task_cfg.get("save_freq_steps", 100_000) == 0:
                path = os.path.join(
                    self.task_cfg.get("output_dir", "results"),
                    f"policy_step{t}.pth"
                )
                torch.save(self.policy.state_dict(), path)

        # Sauvegarde finale
        final_path = os.path.join(
            self.task_cfg.get("output_dir", "results"),
            "policy_final.pth"
        )
        torch.save(self.policy.state_dict(), final_path)
        print(f"Training complete. Policy saved at {final_path}")



    def play(self, episodes=1):
        """
        Lance la simulation MuJoCo en temps réel pour visualiser la
        policy entraînée. La fenêtre s'ouvre et montre la main
        saisir et soulever le cube.
        """
        viewer = mujoco.viewer(self.model, self.data)
        for _ in range(episodes):
            obs = self.task.reset()
            done = False
            while not done:
                # Génère action
                with torch.no_grad():
                    a_t, _ = self.policy.sample(
                        torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    )
                action = a_t.cpu().numpy()
                # Step sim + render
                obs, _, done, _ = self.task.step(action)
                viewer.render()


# -----------------------------------------------------------------------------
# 7) MAIN: orchestration
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Combiner les deux MJCF en un seul
    combined_xml = build_combined_xml(
        args.body_xml, args.fingers_xml, args.output_dir
    )

    # Créer et lancer le trainer SAC
    trainer = SACTrainer(cfg, combined_xml)
    trainer.train()
    trainer.play(episodes=10)

if __name__ == "__main__":
    main()
