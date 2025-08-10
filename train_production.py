#!/usr/bin/env python3
"""
Script d'entraînement de production avec monitoring avancé
"""

import os
import warnings
import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import time
import json

# Configuration
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

class ProgressMonitorCallback(BaseCallback):
    """Callback pour monitorer les progrès d'entraînement"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.stats = {"episodes": 0, "total_steps": 0, "start_time": time.time()}
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Statistiques d'entraînement
            elapsed = time.time() - self.stats["start_time"]
            steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 PROGRÈS - Step {self.n_calls}")
            print(f"⏱️  Temps écoulé: {elapsed/60:.1f}min")
            print(f"🔄 Steps/sec: {steps_per_sec:.1f}")
            
            # Tester l'environnement actuel
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                obs, _ = env.reset()
                
                total_reward = 0
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, term, trunc, info = env.step(action)
                    total_reward += reward
                
                avg_reward = total_reward / 10
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    print(f"🎯 Nouveau meilleur reward: {avg_reward:.3f}")
                
                print(f"📈 Reward moyen (10 steps): {avg_reward:.3f}")
                print(f"🏆 Meilleur reward: {self.best_reward:.3f}")
                
                # Sauvegarder les stats
                self.stats.update({
                    "current_step": self.n_calls,
                    "avg_reward": avg_reward,
                    "best_reward": self.best_reward,
                    "elapsed_minutes": elapsed/60
                })
                
                with open("/workspace/results/training_stats.json", "w") as f:
                    json.dump(self.stats, f, indent=2)
        
        return True

class BalancedGraspEnv(gym.Env):
    """Environnement équilibré optimisé pour production"""
    
    def __init__(self):
        super().__init__()
        
        model_path = "/workspace/results/g1_combined_balanced.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Actuateurs droits
        self.right_actuator_ids = []
        for i in range(self.model.nu):
            if 'right' in self.model.actuator(i).name:
                self.right_actuator_ids.append(i)
        
        # Espaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.right_actuator_ids),), 
            dtype=np.float32
        )
        
        obs_size = 3 + 7 + 8  # cube + right_arm + right_fingers
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Paramètres optimisés
        self.ARM_SCALE = 0.6     # Légèrement augmenté
        self.FINGER_SCALE = 0.4  # Légèrement augmenté
        
        # IDs
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
                 self.right_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        
        # Compteurs pour monitoring
        self.step_count = 0
        self.episode_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Position cube avec légère variation pour diversité
        base_pos = [0.5, 0.2, 1.0]
        variation = np.random.uniform(-0.05, 0.05, 3)
        self.data.qpos[0:3] = base_pos + variation
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.episode_count += 1
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.data.ctrl[:] = 0.0
        action = np.clip(action, -1.0, 1.0)
        
        # Distance adaptative
        cube_pos = self.data.xpos[self.cube_id]
        if self.right_hand_id >= 0:
            hand_pos = self.data.xpos[self.right_hand_id]
            distance = np.linalg.norm(cube_pos - hand_pos)
        else:
            distance = 1.0
        
        distance_factor = min(1.0, distance / 0.3)
        
        # Appliquer actions
        arm_actions = action[:7]
        finger_actions = action[7:] if len(action) > 7 else []
        
        for i, actuator_id in enumerate(self.right_actuator_ids):
            if i < 7 and i < len(arm_actions):
                scaled = arm_actions[i] * self.ARM_SCALE * distance_factor
                self.data.ctrl[actuator_id] = scaled
            elif i >= 7:
                finger_idx = i - 7
                if finger_idx < len(finger_actions):
                    scaled = finger_actions[finger_idx] * self.FINGER_SCALE
                    self.data.ctrl[actuator_id] = scaled
        
        mujoco.mj_step(self.model, self.data)
        
        reward = self._calculate_reward()
        obs = self._get_observation()
        
        self.step_count += 1
        terminated = self.step_count >= 1000  # Épisodes de 1000 steps
        
        info = {
            "distance": distance,
            "episode": self.episode_count,
            "step": self.step_count
        }
        
        return obs, reward, terminated, False, info
    
    def _calculate_reward(self):
        """Reward optimisé pour apprentissage progressif"""
        
        cube_pos = self.data.xpos[self.cube_id]
        
        if self.right_hand_id >= 0:
            hand_pos = self.data.xpos[self.right_hand_id]
            distance = np.linalg.norm(cube_pos - hand_pos)
        else:
            distance = 2.0
        
        # 1. Reward de proximité (progressif)
        if distance < 0.1:
            distance_reward = 10.0  # Très proche
        elif distance < 0.2:
            distance_reward = 5.0   # Proche
        elif distance < 0.5:
            distance_reward = -distance * 5.0  # Linéaire
        else:
            distance_reward = -10.0  # Trop loin
        
        # 2. Reward de stabilité
        cube_vel = np.linalg.norm(self.data.cvel[self.cube_id][:3])
        stability_reward = -min(cube_vel, 5.0)  # Pénaliser vitesse excessive
        
        # 3. Bonus de hauteur
        height_reward = max(0, cube_pos[2] - 0.9) * 2.0
        
        # 4. Bonus de contact
        contact_reward = min(self.data.ncon * 2.0, 10.0)
        
        return distance_reward + stability_reward + height_reward + contact_reward
    
    def _get_observation(self):
        cube_pos = self.data.xpos[self.cube_id][:3]
        right_arm_pos = self.data.qpos[13:20]
        right_finger_pos = self.data.qpos[28:36]
        
        obs = np.concatenate([cube_pos, right_arm_pos, right_finger_pos])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    
    def render(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        pass

def main():
    print("🚀 ENTRAÎNEMENT DE PRODUCTION")
    print("=" * 50)
    
    # Créer environnement
    env = BalancedGraspEnv()
    
    # Test initial
    print("🧪 Test initial de l'environnement...")
    obs, _ = env.reset()
    
    rewards = []
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        if i % 5 == 0:
            print(f"  Step {i}: reward={reward:.2f}, distance={info.get('distance', 0):.3f}")
    
    avg_reward = np.mean(rewards)
    print(f"✅ Reward moyen initial: {avg_reward:.2f}")
    
    if avg_reward > -50:  # Seuil raisonnable
        print("🎯 Environnement OK - Démarrage entraînement long...")
        
        # Configuration TD3 optimisée
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.25 * np.ones(env.action_space.shape[0])
        )
        
        model = TD3(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=500_000,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1
        )
        
        # Callback de monitoring
        callback = ProgressMonitorCallback(check_freq=2000)
        
        print("🎓 DÉMARRAGE ENTRAÎNEMENT COMPLET...")
        print("📊 Objectif: 100,000 steps")
        print("💾 Modèle sera sauvegardé toutes les 10,000 steps")
        
        # Entraînement avec sauvegarde intermédiaire
        for phase in range(10):
            print(f"\n🔄 Phase {phase+1}/10 - Steps {phase*10000} à {(phase+1)*10000}")
            
            model.learn(
                total_timesteps=10_000,
                callback=callback,
                reset_num_timesteps=False
            )
            
            # Sauvegarde intermédiaire
            model.save(f"/workspace/results/model_phase_{phase+1}")
            print(f"💾 Modèle sauvegardé: phase_{phase+1}")
        
        # Sauvegarde finale
        model.save("/workspace/results/final_balanced_model")
        print("🎉 ENTRAÎNEMENT TERMINÉ!")
        print("📁 Modèle final: /workspace/results/final_balanced_model")
        
    else:
        print("❌ Environnement instable - rewards trop négatifs")
        print("🔧 Ajustez les paramètres de reward ou les échelles d'action")
    
    env.close()

if __name__ == "__main__":
    main()