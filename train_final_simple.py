#!/usr/bin/env python3
"""
Script d'entraînement final ultra-simple - Sans OpenGL
"""

import os
import warnings
import numpy as np
import gymnasium as gym

# Configuration avant imports
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

class FinalGraspEnv(gym.Env):
    """Environnement final ultra-simple et stable"""
    
    def __init__(self):
        super().__init__()
        
        # Charger le modèle équilibré
        model_path = "/workspace/results/g1_combined_balanced.xml"
        print(f"📁 Chargement: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"✅ Modèle chargé - Timestep: {self.model.opt.timestep}")
        
        # Actuateurs droits
        self.right_actuators = []
        for i in range(self.model.nu):
            if 'right' in self.model.actuator(i).name:
                self.right_actuators.append(i)
        
        print(f"🎛️ {len(self.right_actuators)} actuateurs droits")
        
        # Espaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.right_actuators),), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,),  # 3 cube + 7 bras + 8 doigts
            dtype=np.float32
        )
        
        # IDs
        self.cube_id = 2  # Corps cube
        self.hand_id = 16  # Corps main droite
        
        # Paramètres
        self.ARM_SCALE = 0.6
        self.FINGER_SCALE = 0.4
        
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Position cube avec légère variation
        base_pos = [0.5, 0.2, 1.0]
        noise = np.random.uniform(-0.03, 0.03, 3)
        self.data.qpos[0:3] = base_pos + noise
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[:] = 0.0
        action = np.clip(action, -1.0, 1.0)
        
        # Distance adaptative
        cube_pos = self.data.xpos[self.cube_id]
        hand_pos = self.data.xpos[self.hand_id]
        distance = np.linalg.norm(cube_pos - hand_pos)
        distance_factor = min(1.0, distance / 0.3)
        
        # Appliquer actions
        arm_actions = action[:7]
        finger_actions = action[7:15] if len(action) > 7 else []
        
        # Bras (premiers 7 actuateurs droits)
        for i in range(min(7, len(arm_actions))):
            if i < len(self.right_actuators):
                actuator_id = self.right_actuators[i]
                scaled = arm_actions[i] * self.ARM_SCALE * distance_factor
                self.data.ctrl[actuator_id] = scaled
        
        # Doigts (actuateurs droits restants)
        for i in range(len(finger_actions)):
            actuator_idx = 7 + i
            if actuator_idx < len(self.right_actuators):
                actuator_id = self.right_actuators[actuator_idx]
                scaled = finger_actions[i] * self.FINGER_SCALE
                self.data.ctrl[actuator_id] = scaled
        
        mujoco.mj_step(self.model, self.data)
        
        # Reward simple mais efficace
        reward = self._calc_reward(distance)
        
        obs = self._get_obs()
        
        self.step_count += 1
        terminated = self.step_count >= 500
        
        return obs, reward, terminated, False, {}
    
    def _calc_reward(self, distance):
        """Reward simple et efficace"""
        cube_pos = self.data.xpos[self.cube_id]
        
        # 1. Reward distance
        if distance < 0.1:
            dist_reward = 20.0
        elif distance < 0.2:
            dist_reward = 10.0
        else:
            dist_reward = -distance * 8.0
        
        # 2. Reward contacts
        contact_reward = min(self.data.ncon * 3.0, 15.0)
        
        # 3. Reward hauteur
        height_reward = max(0, cube_pos[2] - 0.9) * 5.0
        
        # 4. Stabilité
        cube_vel = np.linalg.norm(self.data.cvel[self.cube_id][:3])
        stability_reward = -min(cube_vel, 3.0)
        
        return dist_reward + contact_reward + height_reward + stability_reward
    
    def _get_obs(self):
        """Observation sécurisée"""
        cube_pos = self.data.xpos[self.cube_id][:3]
        right_arm = self.data.qpos[13:20]
        right_fingers = self.data.qpos[28:36]
        
        obs = np.concatenate([cube_pos, right_arm, right_fingers])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    def render(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        pass

def main():
    print("🤖 ENTRAÎNEMENT ROBOTIQUE FINAL")
    print("=" * 40)
    
    # Test environnement
    print("🧪 Test environnement...")
    env = FinalGraspEnv()
    
    obs, _ = env.reset()
    print(f"✅ Reset OK - Obs: {obs.shape}")
    
    # Test rewards
    rewards = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        if i % 3 == 0:
            print(f"Step {i}: reward={reward:.2f}")
    
    avg_reward = np.mean(rewards)
    print(f"✅ Reward moyen: {avg_reward:.2f}")
    
    if avg_reward > -50:
        print("🎯 Environnement OK - Démarrage TD3...")
        
        # Configuration TD3
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.2 * np.ones(env.action_space.shape[0])
        )
        
        model = TD3(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=200_000,
            gamma=0.99,
            tau=0.005,
            verbose=1
        )
        
        print("🎓 ENTRAÎNEMENT 20K STEPS...")
        
        # Entraînement simple
        model.learn(total_timesteps=20_000)
        
        # Sauvegarde
        model_path = "/workspace/results/final_robot_model"
        model.save(model_path)
        
        print("✅ ENTRAÎNEMENT TERMINÉ!")
        print(f"📁 Modèle: {model_path}")
        
        # Test final
        print("🎯 Test final...")
        obs, _ = env.reset()
        
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            print(f"Test {i}: reward={reward:.2f}")
        
        print("🎉 ROBOT ENTRAÎNÉ AVEC SUCCÈS!")
        
    else:
        print("❌ Environnement instable")
    
    env.close()

if __name__ == "__main__":
    main()