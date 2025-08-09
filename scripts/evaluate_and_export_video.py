#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import imageio
import imageio_ffmpeg

# Prefer headless rendering backends
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "osmesa"))

try:
    from google.colab import files as gcolab_files  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    gcolab_files = None

from envs.ultra_robust_grasp_env import make_ultra_robust_grasp_env
from utils.alternative_video_recorder import AlternativeVideoRecorder

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and export video from UltraRobustGraspEnv")
    p.add_argument("--timesteps", type=int, default=600)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("videos/custom_robot_eval.mp4"))
    p.add_argument("--model_path", type=str, default="/workspace/results/g1_combined.xml")
    p.add_argument("--no_viewer", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = make_ultra_robust_grasp_env(
        model_path=args.model_path,
        render_mode="rgb_array",
        enable_curriculum=False,
        enable_mujoco_viewer=not args.no_viewer,
    )

    obs, _ = env.reset(seed=args.seed)

    # Try direct video writing via imageio
    try:
        writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8)
        try:
            for t in range(args.timesteps):
                action = np.random.uniform(-0.05, 0.05, size=env.action_space.shape[0]).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    writer.append_data(frame)
                if terminated or truncated:
                    break
        finally:
            writer.close()
            env.close()
        print(f"✅ Video saved to: {args.out.resolve()}")
    except Exception as e:
        # Fallback: pickle-based frames dump
        print(f"⚠️ Falling back to alternative recorder due to: {e}")
        recorder = AlternativeVideoRecorder(output_dir=str(args.out.parent), fps=args.fps)
        path, info = recorder.record_episode(env, agent=None, max_steps=args.timesteps, render_mode="rgb_array")
        env.close()
        print(f"✅ Frames saved (pickle): {path}")

    if IN_COLAB and gcolab_files is not None:
        try:
            gcolab_files.download(str(args.out))
        except Exception:
            pass


if __name__ == "__main__":
    main()