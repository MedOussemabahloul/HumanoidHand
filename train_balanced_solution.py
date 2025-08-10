#!/usr/bin/env python3
"""
Script d'entraînement équilibré - Stabilité + Performance
"""
import os
import warnings
import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

# Configuration pour éviter les problèmes OpenGL
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

class BalancedGraspEnv(gym.Env):
 """Environnement équilibré pour saisie - stable mais performant"""
 
 def __init__(self):
     super().__init__()
     
     # Charger le modèle équilibré
     model_path = "/home/oussema/Documents/project/results/g1_combined_balanced.xml"
     print(f"📁 Chargement du modèle équilibré: {model_path}")
     
     try:
         self.model = mujoco.MjModel.from_xml_path(model_path)
         self.data = mujoco.MjData(self.model)
         print(f"✅ Modèle chargé - Timestep: {self.model.opt.timestep}")
     except Exception as e:
         print(f"❌ Erreur modèle: {e}")
         raise
     
     # Identifier les actuateurs droits (comme dans le notebook)
     self.right_actuator_ids = []
     for i in range(self.model.nu):
         actuator_name = self.model.actuator(i).name
         if 'right' in actuator_name:
             self.right_actuator_ids.append(i)
     
     print(f"🎛️ Actuateurs droits: {self.right_actuator_ids}")
     
     # Espaces d'action et observation
     self.action_space = gym.spaces.Box(
         low=-1.0, high=1.0, 
         shape=(len(self.right_actuator_ids),), 
         dtype=np.float32
     )
     
     # Observation: positions cube + bras droit + doigts droits
     obs_size = 3 + 7 + 8  # cube_pos + right_arm + right_fingers
     self.observation_space = gym.spaces.Box(
         low=-np.inf, high=np.inf,
         shape=(obs_size,),
         dtype=np.float32
     )
     
     # Paramètres d'échelle (équilibrés)
     self.ARM_SCALE = 0.5    # Réduit par rapport au notebook (0.8)
     self.FINGER_SCALE = 0.3  # Réduit par rapport au notebook (0.5)
     
     # IDs des corps
     self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
     self.right_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
     
     print(f"🎯 Cube ID: {self.cube_id}, Main droite ID: {self.right_hand_id}")
 
 def reset(self, seed=None, options=None):
     """Reset avec position fixe du cube (comme le notebook)"""
     super().reset(seed=seed)
     
     # Reset de la simulation
     mujoco.mj_resetData(self.model, self.data)
     
     # Position fixe du cube (comme le notebook)
     self.data.qpos[0:3] = [0.5, 0.2, 1.0]  # x, y, z
     self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion
     
     # Reset des contrôles (IMPORTANT pour stabilité)
     self.data.ctrl[:] = 0.0
     
     # Forward pour mise à jour
     mujoco.mj_forward(self.model, self.data)
     
     obs = self._get_observation()
     info = {"cube_pos": self.data.xpos[self.cube_id].copy()}
     
     return obs, info
 
 def step(self, action):
     """Step avec échelle d'action adaptative"""
     
     # Reset des contrôles à chaque step (comme le notebook)
     self.data.ctrl[:] = 0.0
     
     # Appliquer les actions avec échelle
     action = np.clip(action, -1.0, 1.0)
     
     # Séparer bras et doigts
     arm_actions = action[:7]    # 7 DOFs bras droit
     finger_actions = action[7:] # 8 DOFs doigts droits
     
     # Échelle adaptative basée sur la distance (comme le notebook)
     cube_pos = self.data.xpos[self.cube_id]
     hand_pos = self.data.xpos[self.right_hand_id]
     distance = np.linalg.norm(cube_pos - hand_pos)
     
     # Plus proche = mouvements plus doux
     distance_factor = min(1.0, distance / 0.3)
     
     # Appliquer aux actuateurs droits
     arm_start = 0
     finger_start = 7
     
     for i, actuator_id in enumerate(self.right_actuator_ids):
         if i < 7:  # Actuateurs de bras
             scaled_action = arm_actions[i] * self.ARM_SCALE * distance_factor
             self.data.ctrl[actuator_id] = scaled_action
         else:  # Actuateurs de doigts
             finger_idx = i - 7
             if finger_idx < len(finger_actions):
                 scaled_action = finger_actions[finger_idx] * self.FINGER_SCALE
                 self.data.ctrl[actuator_id] = scaled_action
     
     # Simulation step
     mujoco.mj_step(self.model, self.data)
     
     # Calcul du reward (basé sur le notebook)
     reward = self._calculate_reward()
     
     # Observation
     obs = self._get_observation()
     
     # Condition de terminaison (épisode long)
     terminated = False
     truncated = False
     
     info = {
         "cube_pos": self.data.xpos[self.cube_id].copy(),
         "distance": distance,
         "reward_components": self._get_reward_components()
     }
     
     return obs, reward, terminated, truncated, info
 
 def _calculate_reward(self):
     """Calcul du reward basé sur le notebook fonctionnel"""
     
     cube_pos = self.data.xpos[self.cube_id]
     hand_pos = self.data.xpos[self.right_hand_id]
     
     # 1. Reward de distance (principal)
     distance = np.linalg.norm(cube_pos - hand_pos)
     distance_reward = -distance * 10.0  # Encourager la proximité
     
     # 2. Reward de vitesse du cube (stabilité)
     cube_vel = np.linalg.norm(self.data.cvel[self.cube_id][:3])
     velocity_reward = -cube_vel * 2.0  # Pénaliser les mouvements brusques
     
     # 3. Reward de contact (bonus si contact)
     contact_reward = 0.0
     for i in range(self.data.ncon):
         contact = self.data.contact[i]
         if contact.geom1 == self.cube_id or contact.geom2 == self.cube_id:
             contact_reward += 5.0  # Bonus pour contact
     
     # 4. Reward de hauteur (garder le cube en l'air)
     height_reward = max(0, cube_pos[2] - 0.8) * 5.0
     
     # Total
     total_reward = distance_reward + velocity_reward + contact_reward + height_reward
     
     return total_reward
 
 def _get_reward_components(self):
     """Retourne les composants du reward pour debug"""
     cube_pos = self.data.xpos[self.cube_id]
     hand_pos = self.data.xpos[self.right_hand_id]
     distance = np.linalg.norm(cube_pos - hand_pos)
     cube_vel = np.linalg.norm(self.data.cvel[self.cube_id][:3])
     
     return {
         "distance": distance,
         "cube_velocity": cube_vel,
         "cube_height": cube_pos[2],
         "contacts": self.data.ncon
     }
 
 def _get_observation(self):
     """Observation sécurisée"""
     
     # Position du cube
     cube_pos = self.data.xpos[self.cube_id][:3]
     
     # Positions des joints du bras droit (DOFs 13-19)
     right_arm_pos = self.data.qpos[13:20]
     
     # Positions des doigts droits (DOFs 28-35)
     right_finger_pos = self.data.qpos[28:36]
     
     # Combiner
     obs = np.concatenate([cube_pos, right_arm_pos, right_finger_pos])
     
     # Sécurité NaN/Inf
     obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
     
     return obs.astype(np.float32)
 
 def render(self):
     return np.zeros((480, 640, 3), dtype=np.uint8)
 
 def close(self):
     pass

def main():
 print('🚀 DÉMARRAGE ENTRAÎNEMENT ÉQUILIBRÉ')
 print('=' * 50)
 
 # Créer l'environnement
 env = BalancedGraspEnv()
 
 # Test rapide de l'environnement
 print('🧪 Test rapide de l\'environnement...')
 obs, info = env.reset()
 print(f'✅ Reset OK - Obs shape: {obs.shape}')
 
 # Test de quelques steps
 total_reward = 0
 for i in range(10):
     action = env.action_space.sample()
     obs, reward, term, trunc, info = env.step(action)
     total_reward += reward
     if i % 3 == 0:
         print(f'  Step {i}: reward={reward:.3f}, distance={info["distance"]:.3f}')
 
 print(f'✅ Test terminé - Reward total: {total_reward:.3f}')
 
 if total_reward > -1000:  # Si pas trop négatif
     print('🎯 Environnement fonctionne - démarrage TD3...')
     
     # Configuration TD3 (comme le notebook)
     action_noise = NormalActionNoise(
         mean=np.zeros(env.action_space.shape[0]),
         sigma=0.2 * np.ones(env.action_space.shape[0])  # Réduit par rapport au notebook
     )
     
     model = TD3(
         'MlpPolicy',
         env,
         action_noise=action_noise,
         learning_rate=3e-4,
         batch_size=256,
         buffer_size=100_000,  # Réduit pour démarrage plus rapide
         gamma=0.98,
         tau=0.02,
         verbose=1
     )
     
     print('🎓 Démarrage entraînement...')
     model.learn(total_timesteps=10_000)  # Test court
     
     print('✅ Entraînement terminé!')
     model.save('/home/oussema/Documents/project/results/balanced_model')
     
 else:
     print('❌ Environnement instable - rewards trop négatifs')
 
 env.close()

if __name__ == '__main__':
 main()
