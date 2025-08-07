#!/usr/bin/env python3
"""
Master Grasp Env Wrapper
- Live MuJoCo viewer support during training (passive viewer)
- Safe, robust episode video recording to /home/oussema/Documents/project/
- Works with any Gymnasium env exposing .model and .data (MuJoCo)

New file: env_master_grasp_wrapper.py
"""
import os
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym as gym  # fallback if gymnasium not present

# MuJoCo imports guarded for environments without mujoco installed
try:
    import mujoco  # noqa: F401
    from mujoco import viewer as mj_viewer  # safe import style
    MUJOCO_AVAILABLE = True
except Exception:
    MUJOCO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


class LiveRenderVideoEnv(gym.Wrapper):
    """Gym wrapper that:
    - opens a passive MuJoCo viewer and keeps it synced while stepping
    - records episode videos using env.render('rgb_array')
    - writes robust MP4 with codec fallback (mp4v -> avc1)
    """

    def __init__(
        self,
        env: gym.Env,
        live_view: bool = True,
        record_video: bool = True,
        video_dir: Optional[str] = None,
        video_name_prefix: str = "episode",
        frame_size: Tuple[int, int] = (640, 480),
    ):
        super().__init__(env)
        self.live_view = live_view and MUJOCO_AVAILABLE
        self.record_video = record_video and CV2_AVAILABLE
        self.frame_size = frame_size
        base_dir = "/home/oussema/Documents/project/master_results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_dir = video_dir or os.path.join(base_dir, "videos_" + timestamp)
        self.video_name_prefix = video_name_prefix

        # Runtime state
        self._viewer = None
        self._video_writer = None
        self._current_video_path = None
        self._frames_written = 0
        self._episode_idx = 0

        # Create dirs if possible
        try:
            os.makedirs(self.video_dir, exist_ok=True)
        except Exception:
            pass

    def _ensure_viewer(self):
        if not self.live_view or not MUJOCO_AVAILABLE:
            return
        if self._viewer is None:
            # Require underlying env to expose model/data
            model = getattr(self.env, "model", None)
            data = getattr(self.env, "data", None)
            if model is not None and data is not None:
                try:
                    self._viewer = mj_viewer.launch_passive(model, data)
                except Exception:
                    self._viewer = None

    def _sync_viewer(self):
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                pass

    def _open_video(self):
        if not self.record_video or not CV2_AVAILABLE:
            return
        try:
            name = f"{self.video_name_prefix}_{self._episode_idx:05d}.mp4"
            path = os.path.join(self.video_dir, name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, 30, self.frame_size)
            if not writer.isOpened():
                # fallback to avc1
                fourcc2 = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(path, fourcc2, 30, self.frame_size)
            if not writer.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir l'écrivain vidéo: {path}")
            self._video_writer = writer
            self._current_video_path = path
            self._frames_written = 0
        except Exception:
            self._video_writer = None
            self._current_video_path = None
            self._frames_written = 0

    def _close_video(self):
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
        self._video_writer = None
        self._current_video_path = None
        self._frames_written = 0

    def _write_frame(self):
        if self._video_writer is None or not CV2_AVAILABLE:
            return
        try:
            frame = None
            # Prefer explicit mode arg if supported
            try:
                frame = self.env.render(mode="rgb_array")  # type: ignore[arg-type]
            except TypeError:
                # Some envs use render_mode attribute
                frame = self.env.render()
            if frame is None:
                return
            if frame.shape[:2] != self.frame_size[::-1]:
                frame = cv2.resize(frame, self.frame_size)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            self._video_writer.write(frame_bgr)
            self._frames_written += 1
        except Exception:
            # do not fail training on frame errors
            pass

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        # Prepare viewer and video per episode
        self._ensure_viewer()
        self._close_video()
        self._open_video()
        self._write_frame()
        return obs, info

    def step(self, action):  # type: ignore[override]
        out = self.env.step(action)
        # Sync viewer and record
        self._sync_viewer()
        self._write_frame()
        # Close video on episode end
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            if terminated or truncated:
                self._episode_idx += 1
                self._close_video()
        else:
            # gym classic API
            obs, reward, done, info = out
            if done:
                self._episode_idx += 1
                self._close_video()
        return out

    def close(self):  # type: ignore[override]
        self._close_video()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        try:
            self.env.close()
        except Exception:
            pass

    @property
    def current_video_path(self) -> Optional[str]:
        return self._current_video_path

    @property
    def frames_written(self) -> int:
        return self._frames_written