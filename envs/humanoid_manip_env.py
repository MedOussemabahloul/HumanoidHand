import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_resetData
from mujoco import Renderer

class HumanoidManipEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 xml_path: str = "assets/scenes/complete_scene.xml",
                 render_mode: str = None,
                 width: int = 640,
                 height: int = 480):
        super().__init__()
        print(f"[DEBUG] Loading MuJoCo model from: {xml_path}")
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.viewer = None
        self.renderer = None

        n_act = self.model.nu
        n_obs = self.model.nq + self.model.nv
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        self.target_pos = np.array([0.0, 0.0, 0.05], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.data.qpos[:self.model.nq] += 0.01 * np.random.randn(self.model.nq)
        self.data.ctrl[:] = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action.astype(float)
        mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = -np.linalg.norm(self.data.qpos[self.model.nq-3:self.model.nq] - self.target_pos)
        done = False
        info = {}
        return obs, float(reward), done, False, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def render(self, mode=None):
        mode = mode or self.render_mode or "human"
        if mode == "human":
            if self.viewer is None:
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
            self.viewer.sync()
        elif mode == "rgb_array":
            if self.renderer is None:
                self.renderer = Renderer(self.model, width=self.width, height=self.height)
            self.renderer.update_scene(self.data)
            frame = self.renderer.render()
            return frame
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.renderer = None
