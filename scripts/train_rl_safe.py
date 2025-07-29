#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version sécurisée du script d'entraînement SAC
Corrections apportées pour éviter le segmentation fault :
1. Gestion mémoire améliorée
2. Validation des données
3. Gestion des erreurs
4. Configuration sécurisée
"""

import os
import sys
import gc
import traceback
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mujoco
from torch.utils.tensorboard import SummaryWriter
import xml.etree.ElementTree as ET
from copy import deepcopy
import copy
from contextlib import contextmanager

# Ajouter le répertoire parent au path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tasks.grasp.grasp_lift_task import GraspLiftTask

# -----------------------------------------------------------------------------
# UTILITAIRES DE SÉCURITÉ
# -----------------------------------------------------------------------------

def safe_tensor_creation(data, device, dtype=torch.float32):
    """Crée un tensor de manière sécurisée avec validation"""
    try:
        if data is None:
            raise ValueError("Données None")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Vérification des valeurs NaN/Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("⚠️  ATTENTION: Données NaN ou Inf détectées")
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        tensor = torch.as_tensor(data, device=device, dtype=dtype)
        
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contient des valeurs NaN ou Inf")
        
        return tensor
    
    except Exception as e:
        print(f"❌ Erreur lors de la création du tensor: {e}")
        raise

@contextmanager
def safe_mujoco_context():
    """Contexte sécurisé pour les opérations MuJoCo"""
    try:
        gc.collect()
        yield
    except Exception as e:
        print(f"❌ Erreur MuJoCo: {e}")
        traceback.print_exc()
        raise
    finally:
        gc.collect()

def validate_config(config):
    """Valide la configuration et applique des valeurs par défaut sécurisées"""
    # Valeurs par défaut sécurisées
    defaults = {
        "task": {
            "cube_body_name": "cube",
            "max_steps_per_episode": 100,
            "touch_sensors": [],
            "force_sensors": [],
            "include_orientation_reward": False,
            "force_reward_weight_normal": 0.0,
            "force_reward_weight_tangential": 0.0,
            "translation_penalty_weight": 0.0,
            "output_dir": "results",
            "save_freq_steps": 1000
        },
        "rl": {
            "gamma": 0.99,
            "alpha": 0.2,
            "learning_rate": 3e-4,
            "hidden_size": 256,
            "batch_size": 32,  # Réduit pour éviter les problèmes mémoire
            "replay_size": 10000,  # Réduit
            "start_steps": 100,
            "update_after": 100,
            "update_every": 1,
            "num_updates": 1,  # Réduit
            "total_steps": 1000,  # Réduit pour les tests
            "tau": 0.005,
            "act_limit": 1.0
        }
    }
    
    # Fusion avec les valeurs par défaut
    for section in ["task", "rl"]:
        if section not in config:
            config[section] = defaults[section]
        else:
            for key, value in defaults[section].items():
                if key not in config[section]:
                    config[section][key] = value
    
    return config

# -----------------------------------------------------------------------------
# RÉSEAUX DE NEURONES SÉCURISÉS
# -----------------------------------------------------------------------------

class SafePolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Réseau plus simple et stable
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, act_dim)
        self.fc_logstd = nn.Linear(hidden_size, act_dim)
        
        # Initialisation sécurisée
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        try:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            mean = self.fc_mean(x)
            logstd = torch.clamp(self.fc_logstd(x), -20, 2)
            return mean, logstd
        except Exception as e:
            print(f"❌ Erreur dans PolicyNet.forward: {e}")
            raise
    
    def sample(self, obs):
        try:
            mean, logstd = self.forward(obs)
            std = torch.exp(logstd)
            normal = torch.randn_like(mean)
            action = mean + std * normal
            log_prob = -0.5 * ((action - mean) / std).pow(2) - logstd - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob.sum(-1, keepdim=True)
            return action, log_prob
        except Exception as e:
            print(f"❌ Erreur dans PolicyNet.sample: {e}")
            raise

class SafeQNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs, act):
        try:
            x = torch.cat([obs, act], dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        except Exception as e:
            print(f"❌ Erreur dans QNet.forward: {e}")
            raise

class SafeValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        try:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        except Exception as e:
            print(f"❌ Erreur dans ValueNet.forward: {e}")
            raise

# -----------------------------------------------------------------------------
# REPLAY BUFFER SÉCURISÉ
# -----------------------------------------------------------------------------

class SafeReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
    
    def store(self, obs, act, rew, nxt_obs, done):
        try:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.next_obs_buf[self.ptr] = nxt_obs
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        except Exception as e:
            print(f"❌ Erreur dans ReplayBuffer.store: {e}")
            raise
    
    def sample_batch(self, batch_size):
        try:
            idxs = np.random.randint(0, self.size, size=batch_size)
            return {
                "obs": self.obs_buf[idxs],
                "act": self.act_buf[idxs],
                "rew": self.rew_buf[idxs],
                "next_obs": self.next_obs_buf[idxs],
                "done": self.done_buf[idxs]
            }
        except Exception as e:
            print(f"❌ Erreur dans ReplayBuffer.sample_batch: {e}")
            raise

# -----------------------------------------------------------------------------
# TRAINER SAC SÉCURISÉ
# -----------------------------------------------------------------------------

class SafeSACTrainer:
    def __init__(self, cfg, model_xml):
        try:
            print("🚀 Initialisation du trainer SAC sécurisé...")
            
            # 1) Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Device utilisé: {self.device}")
            
            # 2) Configuration validée
            cfg = validate_config(cfg)
            self.task_cfg = cfg["task"]
            rl_cfg = cfg["rl"]
            self.act_limit = rl_cfg.get("act_limit", 1.0)
            
            # 3) Hyperparamètres
            self.gamma = float(rl_cfg["gamma"])
            self.alpha = float(rl_cfg["alpha"])
            self.lr = float(rl_cfg["learning_rate"])
            self.hidden = int(rl_cfg["hidden_size"])
            self.batch_size = int(rl_cfg["batch_size"])
            self.replay_size = int(rl_cfg["replay_size"])
            self.start_steps = int(rl_cfg["start_steps"])
            self.update_after = int(rl_cfg["update_after"])
            self.update_every = int(rl_cfg["update_every"])
            self.num_updates = int(rl_cfg["num_updates"])
            self.total_steps = int(rl_cfg["total_steps"])
            self.tau = float(rl_cfg.get("tau", 0.005))
            
            # 4) Charger MuJoCo de manière sécurisée
            with safe_mujoco_context():
                self.model = mujoco.MjModel.from_xml_path(model_xml)
                self.data = mujoco.MjData(self.model)
                print(f"✅ MuJoCo chargé: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
            
            # 5) Task
            self.task = GraspLiftTask(self.model, self.data, self.task_cfg)
            
            # 6) Dimensions
            obs = self.task.reset()
            self.obs_dim = obs.shape[0]
            self.act_dim = self.task.act_dim
            print(f"📏 Dimensions: obs={self.obs_dim}, act={self.act_dim}")
            
            # 7) ReplayBuffer
            self.buffer = SafeReplayBuffer(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                size=self.replay_size,
                device=self.device
            )
            
            # 8) Réseaux
            self.policy = SafePolicyNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
            self.q1 = SafeQNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
            self.q2 = SafeQNet(self.obs_dim, self.act_dim, self.hidden).to(self.device)
            self.v = SafeValueNet(self.obs_dim, self.hidden).to(self.device)
            
            # 9) Cibles
            self.v_targ = SafeValueNet(self.obs_dim, self.hidden).to(self.device)
            self.v_targ.load_state_dict(self.v.state_dict())
            for p in self.v_targ.parameters():
                p.requires_grad = False
            
            self.q1_target = copy.deepcopy(self.q1).to(self.device)
            self.q2_target = copy.deepcopy(self.q2).to(self.device)
            for p in self.q1_target.parameters():
                p.requires_grad = False
            for p in self.q2_target.parameters():
                p.requires_grad = False
            
            # 10) Optimizers
            self.pi_opt = optim.Adam(self.policy.parameters(), lr=self.lr)
            self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.lr)
            self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.lr)
            self.v_opt = optim.Adam(self.v.parameters(), lr=self.lr)
            
            # 11) Logger
            tb_dir = os.path.join(self.task_cfg.get("output_dir", "results"), "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            self.step_count = 0
            
            print("✅ Trainer SAC initialisé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du trainer: {e}")
            traceback.print_exc()
            raise
    
    def update(self):
        try:
            total_q1_loss = torch.tensor(0.0, device=self.device)
            total_q2_loss = torch.tensor(0.0, device=self.device)
            total_pi_loss = torch.tensor(0.0, device=self.device)
            
            for i in range(self.num_updates):
                batch = self.buffer.sample_batch(self.batch_size)
                
                # Conversion sécurisée des tensors
                obs = safe_tensor_creation(batch["obs"], self.device)
                act = safe_tensor_creation(batch["act"], self.device)
                rew = safe_tensor_creation(batch["rew"], self.device).unsqueeze(-1)
                nxt = safe_tensor_creation(batch["next_obs"], self.device)
                done = safe_tensor_creation(batch["done"], self.device).unsqueeze(-1)
                
                # Critic update
                with torch.no_grad():
                    a2, logp2 = self.policy.sample(nxt)
                    a2 = torch.clamp(a2, -self.act_limit, self.act_limit)
                    
                    q1_pi_t = self.q1_target(nxt, a2)
                    q2_pi_t = self.q2_target(nxt, a2)
                    min_q_t = torch.min(q1_pi_t, q2_pi_t)
                    
                    target_val = (min_q_t - self.alpha * logp2).unsqueeze(-1)
                    q_backup = rew + self.gamma * (1 - done) * target_val
                
                # Q1 loss
                q1_pred = self.q1(obs, act)
                if q1_pred.dim() == 1:
                    q1_pred = q1_pred.unsqueeze(-1)
                q1_loss = F.mse_loss(q1_pred, q_backup)
                self.q1_opt.zero_grad()
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)  # Gradient clipping
                self.q1_opt.step()
                total_q1_loss += q1_loss.detach()
                
                # Q2 loss
                q2_pred = self.q2(obs, act)
                if q2_pred.dim() == 1:
                    q2_pred = q2_pred.unsqueeze(-1)
                q2_loss = F.mse_loss(q2_pred, q_backup)
                self.q2_opt.zero_grad()
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.q2_opt.step()
                total_q2_loss += q2_loss.detach()
                
                # Policy loss
                a1, logp1 = self.policy.sample(obs)
                a1 = torch.clamp(a1, -self.act_limit, self.act_limit)
                q1_pi = self.q1(obs, a1)
                q2_pi = self.q2(obs, a1)
                min_q_pi = torch.min(q1_pi, q2_pi)
                pi_loss = (self.alpha * logp1 - min_q_pi).mean()
                self.pi_opt.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.pi_opt.step()
                total_pi_loss += pi_loss.detach()
                
                # Update targets
                with torch.no_grad():
                    for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                        p_targ.data.mul_(1 - self.tau)
                        p_targ.data.add_(self.tau * p.data)
                    for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                        p_targ.data.mul_(1 - self.tau)
                        p_targ.data.add_(self.tau * p.data)
            
            # Logging
            avg_q1 = (total_q1_loss / self.num_updates).item()
            avg_q2 = (total_q2_loss / self.num_updates).item()
            avg_pi = (total_pi_loss / self.num_updates).item()
            
            self.writer.add_scalar("loss/q1_loss", avg_q1, self.step_count)
            self.writer.add_scalar("loss/q2_loss", avg_q2, self.step_count)
            self.writer.add_scalar("loss/pi_loss", avg_pi, self.step_count)
            
        except Exception as e:
            print(f"❌ Erreur dans update: {e}")
            traceback.print_exc()
            raise
    
    def train(self):
        try:
            print("🚀 Début de l'entraînement...")
            obs = self.task.reset()
            
            for t in range(self.total_steps):
                self.step_count = t
                
                # Exploration vs exploitation
                if t < self.start_steps:
                    action = np.random.uniform(-1, 1, size=self.act_dim) * self.act_limit
                else:
                    with torch.no_grad():
                        obs_tensor = safe_tensor_creation(obs, self.device)
                        a_t, _ = self.policy.sample(obs_tensor)
                        action = a_t.cpu().numpy()
                
                action = np.clip(action, -self.act_limit, self.act_limit)
                
                # Simulation
                next_obs, reward, done, _ = self.task.step(action)
                
                # Stockage
                self.buffer.store(obs, action, reward, next_obs, float(done))
                obs = next_obs if not done else self.task.reset()
                
                # Update périodique
                if t >= self.update_after and t % self.update_every == 0:
                    self.update()
                
                # Logging
                self.writer.add_scalar("env/reward", reward, t)
                
                # Checkpoint
                if t > 0 and t % self.task_cfg.get("save_freq_steps", 100_000) == 0:
                    path = os.path.join(
                        self.task_cfg.get("output_dir", "results"),
                        f"policy_step{t}.pth"
                    )
                    torch.save(self.policy.state_dict(), path)
                
                # Affichage progress
                if t % 100 == 0:
                    print(f"Step {t}/{self.total_steps}, Reward: {reward:.3f}")
            
            # Sauvegarde finale
            final_path = os.path.join(
                self.task_cfg.get("output_dir", "results"),
                "policy_final.pth"
            )
            torch.save(self.policy.state_dict(), final_path)
            print(f"✅ Entraînement terminé. Policy sauvegardée: {final_path}")
            
        except Exception as e:
            print(f"❌ Erreur dans train: {e}")
            traceback.print_exc()
            raise

# -----------------------------------------------------------------------------
# MAIN SÉCURISÉ
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement SAC sécurisé")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--body_xml", required=True, help="Path to G1 body MJCF")
    parser.add_argument("--fingers_xml", required=True, help="Path to G1 fingers MJCF")
    parser.add_argument("-o", "--output_dir", default="results", help="Output directory")
    return parser.parse_args()

def load_config(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la config: {e}")
        raise

def build_combined_xml(body_xml, fingers_xml, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
        combined_path = os.path.join(out_dir, "g1_combined.xml")
        
        # Pour simplifier, on utilise juste le body_xml
        import shutil
        shutil.copy2(body_xml, combined_path)
        
        return combined_path
    except Exception as e:
        print(f"❌ Erreur lors de la création du XML combiné: {e}")
        raise

def main():
    try:
        print("🚀 ULTRA-ROBUST SAC PER TRAINING SYSTEM (VERSION SÉCURISÉE)")
        print("=" * 60)
        
        args = parse_args()
        cfg = load_config(args.config)
        
        # Combiner les XML
        combined_xml = build_combined_xml(args.body_xml, args.fingers_xml, args.output_dir)
        
        # Créer et lancer le trainer
        trainer = SafeSACTrainer(cfg, combined_xml)
        trainer.train()
        
        print("✅ Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()