#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasks/grasp/grasp_lift_task.py

Grasp & Lift Task for G1 dual-arm.  
Intègre :
  - contact, lift, force, translation & orientation rewards  
  - usage de high_level_planner for quat/Euler & errors  
  - configuration 100% YAML-driven  
  - documentation ligne à ligne  
"""

import numpy as np
import mujoco
from tasks.planner.high_level_planner import (
    quat_to_euler,
    orientation_error
)

class GraspLiftTask:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        # 1) refs MuJoCo
        self.model, self.data = model, data

        # 2) task params
        self.cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["cube_body_name"])
        self.max_steps = int(config.get("max_steps_per_episode", 500))
        self.step_count = 0

        # 3) tactile sensors
        self.touch_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                  for n in config["touch_sensors"]]        # 4) force sensors (3× per doigt)

        self.force_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                        for n in config["force_sensors"]]
        # 5) orientation target & weight
        self.include_ori = bool(config.get("include_orientation_reward", False))
        if self.include_ori:
            # target euler angles
            self.target_euler = np.array(config["target_cube_euler"], dtype=np.float32)
            self.target_quat  = None
            # convert on first reset
            self.w_ori = float(config.get("orientation_reward_weight", 0.0))

        # 6) reward weights
        self.w_fn = float(config.get("force_reward_weight_normal", 0.0))
        self.w_ft = float(config.get("force_reward_weight_tangential", 0.0))
        self.w_tp = float(config.get("translation_penalty_weight", 0.0))

        # 7) initial XY for translation penalty
        self.init_xy = np.zeros(2, dtype=np.float32)

        # 8) obs & action dims
        self.obs_dim = (model.nq + model.nv + 3 +
                        (4 if self.include_ori else 0) +
                        len(self.touch_ids) + len(self.force_ids))
        self.act_dim = model.nu

        # 9) reset sim
        self.reset()

    def reset(self) -> np.ndarray:
        # a) reset MuJoCo data & forward
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # b) step counter
        self.step_count = 0
        # c) record init cube XY
        pos = self.data.xpos[self.cube_id]
        self.init_xy[:] = pos[:2]
        # d) compute target_quat if needed
        if self.include_ori:
            # euler → quaternion
            from tasks.planner.high_level_planner import euler_to_quat
            self.target_quat = euler_to_quat(self.target_euler)
        return self._get_obs()

    def step(self, action: np.ndarray):
        # 1) inc step
        self.step_count += 1
        # 2) apply action
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        # 3) obs + reward
        obs    = self._get_obs()
        reward = self._compute_reward()
        done   = (self.step_count >= self.max_steps)
        return obs, reward, done, {}

    def _get_obs(self) -> np.ndarray:
        # qpos, qvel
        qpos = self.data.qpos[:self.model.nq].copy()
        qvel = self.data.qvel[:self.model.nv].copy()
        # cube pos
        pos3 = self.data.xpos[self.cube_id].copy()
        # cube ori (quat→euler if needed)
        if self.include_ori:
            quat = self.data.xquat[self.cube_id].copy()
            quat /= np.linalg.norm(quat)
            ori = quat_to_euler(quat)
        else:
            ori = np.empty(0, dtype=np.float32)
        # touch flags
        touches = np.array([self.data.sensordata[i] for i in self.touch_ids],
                           dtype=np.float32)
        # force values
        forces  = np.array([self.data.sensordata[i] for i in self.force_ids],
                           dtype=np.float32)
        return np.concatenate([qpos, qvel, pos3, ori, touches, forces])

    def _compute_reward(self) -> float:
        # --- contact reward ---
        touch_vals = [self.data.sensordata[i] for i in self.touch_ids]
        r_contact = float(any(v>0 for v in touch_vals))
        # --- lift reward (cube height) ---
        z = float(self.data.xpos[self.cube_id][2])
        r_lift = z
        # --- force reward ---
        fv = np.array([self.data.sensordata[i] for i in self.force_ids],
                      dtype=np.float32)
        normals  = fv[0::3]
        tangents = np.abs(np.concatenate([fv[1::3], fv[2::3]]))
        r_force = self.w_fn * normals.mean() + self.w_ft * tangents.mean()
        # --- translation penalty ---
        xy = self.data.xpos[self.cube_id][:2]
        dist = np.linalg.norm(xy - self.init_xy)
        r_trans = - self.w_tp * dist
        # --- orientation reward (optional) ---
        if self.include_ori:
            quat = self.data.xquat[self.cube_id].copy()
            quat /= np.linalg.norm(quat)
            # compute axis-angle error
            axis, angle = orientation_error(quat, self.target_quat)
            r_ori = - self.w_ori * angle
        else:
            r_ori = 0.0
        # total
        return r_contact + r_lift + r_force + r_trans + r_ori
