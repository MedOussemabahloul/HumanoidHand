#!/usr/bin/env python3

import os
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class SimpleRightHandEnv(gym.Env):
    """
    Environnement simple et robuste (main droite uniquement), inspiré du notebook collègue,
    adapté à votre modèle `results/g1_combined.xml` et aux conventions de nommage actuelles.
    - Filtrage des actuateurs main/bras droits via noms `act_right_*`
    - Reset sûr avec positionnement du cube via joint `cube_free`
    - Step avec clamp/filtrage d'actions et contrôle des vitesses
    - Récompense dense similaire (distance/contact/grasp qualitatif)
    - Rendu via env.render('rgb_array')
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, model_path: Optional[str] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.model_path = model_path or "/workspace/results/g1_combined.xml"

        # Charger modèle
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Actuateurs: main/bras droits (par nom d'actuateur 'act_right_*')
        right_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_right_"):
                right_actuators.append(i)
        if len(right_actuators) == 0:
            # Fallback: par joint associé contenant 'right_'
            for i in range(self.model.nu):
                trnid = self.model.actuator_trnid[i, 0]
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, trnid) or ""
                if "right_" in jname:
                    right_actuators.append(i)
        self.right_actuator_ids = np.array(right_actuators, dtype=np.int32)
        if len(self.right_actuator_ids) == 0:
            raise RuntimeError("No right-hand actuators found. Check model/actuator names.")

        # Espace d'action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.right_actuator_ids),), dtype=np.float32)

        # Espace d'observation: qpos+qvel + cube_pos(3) + hand_center(3) + relative(3)
        obs_dim = self.model.nq + self.model.nv + 9
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

        # IDs utiles
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        if self.cube_joint_id < 0:
            raise RuntimeError("Joint 'cube_free' not found. Check the model file.")
        self.cube_qpos_addr = int(self.model.jnt_qposadr[self.cube_joint_id])

        # Sites bout des doigts droits (contacts/distance)
        self.right_tip_sites: List[int] = []
        for i in range(self.model.nsite):
            sname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
            if sname.startswith("right_") and sname.endswith("_tip_site"):
                self.right_tip_sites.append(i)

        # Hyper paramètres simples
        self.max_steps = 500
        self.current_step = 0
        self.velocity_limit = 12.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Placer le cube à une position fixe et orientation identité
        fixed_cube_pos = np.array([0.30, 0.0, 0.05], dtype=np.float64)
        fixed_cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.cube_qpos_addr:self.cube_qpos_addr + 3] = fixed_cube_pos
        self.data.qpos[self.cube_qpos_addr + 3:self.cube_qpos_addr + 7] = fixed_cube_quat

        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Sanitize action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(self.right_actuator_ids),):
            action = np.clip(action, -1.0, 1.0)
            if action.size > len(self.right_actuator_ids):
                action = action[:len(self.right_actuator_ids)]
            elif action.size < len(self.right_actuator_ids):
                pad = np.zeros(len(self.right_actuator_ids) - action.size, dtype=np.float32)
                action = np.concatenate([action, pad])
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros_like(action)

        # Split: heuristique 7 bras + reste doigts si dispo
        arm_count = min(7, len(self.right_actuator_ids))
        arm_action = action[:arm_count]
        finger_action = action[arm_count:]

        # Distances utiles
        cube_pos = self.data.xpos[self.cube_body_id]
        hand_center = self._get_hand_center()
        dist = np.linalg.norm(hand_center - cube_pos)

        # Scales adaptatifs
        ARM_SCALE = 0.3 if dist > 0.08 else 0.15
        FINGER_SCALE = 0.6

        # Reset et appliquer commandes (douces)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.right_actuator_ids[:arm_count]] = np.clip(arm_action * ARM_SCALE, -1.0, 1.0)
        if len(finger_action) > 0:
            self.data.ctrl[self.right_actuator_ids[arm_count:]] = np.clip(finger_action * FINGER_SCALE, -1.0, 1.0)

        # Aide à la fermeture si 2 tip proches du cube
        if dist < 0.07 and self._tip_near_cube_count(threshold=0.06) >= 2:
            idx = self.right_actuator_ids[arm_count:]
            self.data.ctrl[idx] = np.clip(self.data.ctrl[idx] + 0.4, -1.0, 1.0)

        # Step physique avec garde-fous
        qpos_bk = self.data.qpos.copy()
        qvel_bk = self.data.qvel.copy()
        mujoco.mj_step(self.model, self.data)
        if (np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or
                np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel))):
            # rollback et réduire commande
            self.data.qpos[:] = qpos_bk
            self.data.qvel[:] = qvel_bk * 0.5
            self.data.ctrl[:] *= 0.5
            mujoco.mj_step(self.model, self.data)

        # Clamp des vitesses excessives
        vmax = np.max(np.abs(self.data.qvel))
        if vmax > self.velocity_limit:
            self.data.qvel[:] *= (self.velocity_limit / vmax)

        obs = self._get_obs()
        reward = self._compute_reward()
        self.current_step += 1

        # Terminaisons simples
        cube_h = self.data.xpos[self.cube_body_id][2]
        terminated = False
        truncated = self.current_step >= self.max_steps or cube_h < 0.0
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        hand_center = self._get_hand_center()
        relative = cube_pos - hand_center
        base = np.concatenate([self.data.qpos, self.data.qvel])
        obs = np.concatenate([base, cube_pos, hand_center, relative]).astype(np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_hand_center(self) -> np.ndarray:
        if len(self.right_tip_sites) == 0:
            # fallback estimation
            return np.array([0.35, 0.0, 0.10])
        positions = []
        for sid in self.right_tip_sites:
            positions.append(self.data.site_xpos[sid])
        return np.mean(np.array(positions), axis=0)

    def _tip_near_cube_count(self, threshold: float = 0.06) -> int:
        cube = self.data.xpos[self.cube_body_id]
        count = 0
        for sid in self.right_tip_sites:
            d = np.linalg.norm(self.data.site_xpos[sid] - cube)
            if d < threshold:
                count += 1
        return count

    def _detect_contact_with_cube(self) -> bool:
        if self.cube_body_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == self.cube_body_id or b2 == self.cube_body_id:
                other = b2 if b1 == self.cube_body_id else b1
                # Un corps de la main droite contient souvent 'right'
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
                if "right" in name.lower():
                    return True
        return False

    def _compute_reward(self) -> float:
        cube_pos = self.data.xpos[self.cube_body_id]
        hand_center = self._get_hand_center()
        dist = np.linalg.norm(hand_center - cube_pos)
        cube_vel = np.linalg.norm(self.data.cvel[self.cube_body_id])

        touch_count = self._tip_near_cube_count(threshold=0.06)
        contact = self._detect_contact_with_cube()

        # Heuristique qualité de prise
        if touch_count == 0:
            grasp_quality = -1.0
        elif touch_count == 1:
            grasp_quality = 0.1
        elif touch_count == 2:
            grasp_quality = 0.4
        else:
            grasp_quality = 0.9 if cube_vel < 0.05 else 0.5

        reward = 0.0
        reward += 5.0 / (1.0 + 20.0 * dist)         # proximité
        reward += 2.0 if dist < 0.06 else 0.0        # bonus proche
        reward += 10.0 * grasp_quality               # qualité de prise
        reward += 1.0 if contact else 0.0            # léger bonus contact
        reward -= 2.0 * min(1.0, cube_vel)           # pénalité vitesse cube
        reward -= 0.005                               # coût temps

        return float(np.clip(reward, -50.0, 200.0))


def make_simple_right_hand_env(**kwargs) -> SimpleRightHandEnv:
    return SimpleRightHandEnv(**kwargs)