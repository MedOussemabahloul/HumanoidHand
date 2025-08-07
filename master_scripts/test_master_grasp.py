#!/usr/bin/env python3
"""
Smoke test for LiveRenderVideoEnv + CurriculumGraspEnv
- Resets env, runs a few steps with random actions
- Ensures live viewer opens if Mujoco is available
- Writes a short MP4 with codec fallback

New file; minimal and clear.
"""
import os
import sys
import time

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ENV_DIR = "/home/oussema/Documents/project/envs"
if PROJECT_ENV_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ENV_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from envs.curriculum_grasp_env import CurriculumGraspEnv
from env_master_grasp_wrapper import LiveRenderVideoEnv


def main():
    env = CurriculumGraspEnv(render_mode="rgb_array")
    wrapper = LiveRenderVideoEnv(
        env,
        live_view=True,
        record_video=True,
        video_dir="/home/oussema/Documents/project/master_smoke_videos",
        video_name_prefix="smoke",
        frame_size=(640, 480),
    )

    obs, info = wrapper.reset()
    total_r = 0.0
    for step in range(100):
        action = wrapper.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = wrapper.step(action)
        total_r += float(reward)
        time.sleep(0.01)
        if terminated or truncated:
            break

    print(f"✅ Smoke test OK, reward={total_r:.2f}, frames={wrapper.frames_written}")
    print(f"📹 Video path: {wrapper.current_video_path}")
    wrapper.close()


if __name__ == "__main__":
    main()