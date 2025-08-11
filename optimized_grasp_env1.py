"""
🤖 ENVIRONNEMENT GRASPING OPTIMISÉ - INSPIRÉ DU COLLÈGUE
========================================================

Version optimisée qui intègre les meilleures pratiques du collègue
avec corrections des problèmes de stagnation identifiés.

✅ SOLUTIONS APPLIQUÉES:
- Position cube plus accessible [0.15, 0.0, 0.04] 
- Reset contrôles systématique
- Scaling adaptatif efficace
- Système rewards motivant
- Assistance grasping intelligente
"""

import mujoco
from mujoco import MjModel, MjData
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import os

class OptimizedGraspEnv1(gym.Env):
    def __init__(self, model_path=None, render_mode="rgb_array", max_episode_steps=500):
        super().__init__()
        
        # Logger simple
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Modèle XML - utiliser celui qui marche
        if model_path is None:
            model_path = "results/g1_combined.xml"
        
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)

        # Actuators: right side only (EXACTEMENT comme l'ami)
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None and name.startswith("right_"):
                right_actuators.append(i)
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.right_actuator_ids),),
            dtype=np.float32
        )

        self.renderer = mujoco.Renderer(self.model, width=640, height=480)

        obs_dim = self.model.nq + self.model.nv + 9
        self.observation_space = spaces.Box(
            low=-1e10, high=1e10,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = max_episode_steps
        self.success_counter = 0
        self.freeze_timer = 0

        self.logger.info(f"🤖 Environnement initialisé - {len(self.right_actuator_ids)} actuators")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        # Position cube EXACTEMENT comme l'ami
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        fixed_cube_pos = np.array([0.18, 0.0, 0.04])
        start = cube_qpos_addr
        end = cube_qpos_addr + 3
        if end <= len(self.data.qpos):
            self.data.qpos[start:end] = fixed_cube_pos
        else:
            size = len(self.data.qpos) - start
            if size > 0:
                self.data.qpos[start:start + size] = fixed_cube_pos[:size]

        fixed_cube_quat = np.array([1, 0, 0, 0])
        start = cube_qpos_addr + 3
        end = cube_qpos_addr + 7
        if end <= len(self.data.qpos):
            self.data.qpos[start:end] = fixed_cube_quat
        else:
            size = len(self.data.qpos) - start
            if size > 0:
                self.data.qpos[start:start + size] = fixed_cube_quat[:size]

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Split action EXACTEMENT comme l'ami
        arm_action = action[:7]
        finger_action = action[7:]

        # Get positions avec les VRAIS noms du XML
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = self.data.xpos[cube_id]
        palm_pos = self.data.body("right_index_1").xpos
        thumb_pos = self.data.body("right_thumb_1").xpos
        index_pos = self.data.body("right_index_1").xpos
        middle_pos = self.data.body("right_middle_1").xpos

        # Distances
        dist = np.linalg.norm(palm_pos - cube_pos)
        thumb_dist = np.linalg.norm(thumb_pos - cube_pos)
        index_dist = np.linalg.norm(index_pos - cube_pos)
        middle_dist = np.linalg.norm(middle_pos - cube_pos)

        # Contact detection avec les VRAIS noms
        thumb_contact = self._is_touching("cube_geom", "right_thumb_1_geom")
        index_contact = self._is_touching("cube_geom", "right_index_1_geom")
        middle_contact = self._is_touching("cube_geom", "right_middle_1_geom")
        num_contacts = sum([thumb_contact, index_contact, middle_contact])

        # Scale actions EXACTEMENT comme l'ami
        ARM_SCALE = 0.4 if dist > 0.08 else 0.2
        FINGER_SCALE = 0.7

        # Reset controls (CLÉS DU SUCCÈS!)
        self.data.ctrl[:] = 0.0

        # Apply scaled actions EXACTEMENT comme l'ami
        self.data.ctrl[self.right_actuator_ids[:7]] = arm_action * ARM_SCALE
        self.data.ctrl[self.right_actuator_ids[7:]] = finger_action * FINGER_SCALE

        # Grasp assist EXACTEMENT comme l'ami
        if dist < 0.06 and num_contacts >= 2:
            assist_strength = 0.5
            self.data.ctrl[self.right_actuator_ids[7:]] += assist_strength
            self.data.ctrl[self.right_actuator_ids[7:]] = np.clip(
                self.data.ctrl[self.right_actuator_ids[7:]], -1.0, 1.0
            )
            print("🤝 Grasp assist triggered (≥2 fingers touching)")

        # Step simulation
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        self.current_step += 1

        # Termination EXACTEMENT comme l'ami
        done = (
            dist > 0.5
            or cube_pos[2] < 0.01
            or cube_pos[2] > 1.0
            or self.current_step >= self.max_steps
        )

        # Info
        info = {
            'distance': dist,
            'contact_count': num_contacts,
            'cube_velocity': np.linalg.norm(self.data.cvel[cube_id]),
            'total_reward': reward,
            'episode_step': self.current_step
        }

        return obs, reward, done, False, info

    def _compute_reward(self):
        # REWARDS avec les VRAIS noms
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = self.data.xpos[cube_id]
        palm_pos = self.data.body("right_index_1").xpos

        dist = np.linalg.norm(palm_pos - cube_pos)
        cube_vel = np.linalg.norm(self.data.cvel[cube_id])

        # Count how many fingers are touching the cube
        fingers = [
            "right_thumb_1",
            "right_index_1", 
            "right_middle_1"
        ]
        touch_count = sum(self._is_touching(f, "cube") for f in fingers)

        # Grasp quality heuristic EXACTEMENT comme l'ami
        if touch_count == 0:
            grasp_quality = -1.0
        elif touch_count == 1:
            grasp_quality = 0.1
        elif touch_count == 2:
            grasp_quality = 0.4
        else:  # 3+
            grasp_quality = 0.9 if cube_vel < 0.05 else 0.5

        # Reward components EXACTEMENT comme l'ami
        reward = 0
        reward += 5.0 / (1.0 + 20 * dist)
        reward += 2.0 if dist < 0.06 else 0
        reward += 10.0 * grasp_quality
        reward -= 2.0 * min(1.0, cube_vel)
        reward -= 0.005  # time penalty

        return reward

    def _get_obs(self):
        # Observations avec les VRAIS noms
        cube_pos = self.data.body("cube").xpos.copy()
        palm_pos = self.data.body("right_index_1").xpos.copy()
        relative_pos = cube_pos - palm_pos
        base_state = np.concatenate([self.data.qpos, self.data.qvel])
        obs = np.concatenate([base_state, cube_pos, palm_pos, relative_pos])
        expected_dim = self.observation_space.shape[0]
        fixed_obs = np.zeros(expected_dim, dtype=np.float32)
        obs = obs.astype(np.float32)

        fixed_obs[:min(expected_dim, obs.shape[0])] = obs[:min(expected_dim, obs.shape[0])]
        return fixed_obs

    def _is_touching(self, geom1, geom2):
        # Contact detection EXACTEMENT comme l'ami
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom1 in (name1, name2)) and (geom2 in (name1, name2)):
                return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()