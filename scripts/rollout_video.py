#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import imageio
import imageio_ffmpeg
import mujoco
from mujoco import mj_step, Renderer
from mujoco.viewer import launch_passive
from envs.humanoid_manip_env import HumanoidManipEnv

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

def parse_args():
    p = argparse.ArgumentParser(description="Rollout MuJoCo + vidéo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "videos" / "rollout.mp4")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"▶ Rollout MuJoCo : steps={args.steps}, fps={args.fps}, seed={args.seed}")
    print(f"  Vidéo de sortie : {out_path}\n")

    # Charge l'env main réaliste
    env = HumanoidManipEnv(render_mode=None, width=640, height=480)
    obs, _ = env.reset(seed=args.seed)
    viewer = launch_passive(env.model, env.data)
    renderer = Renderer(env.model, width=env.width, height=env.height)

    # Utilise la caméra main_cam
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
    if cam_id < 0:
        raise RuntimeError('Camera "main_cam" not found in the model!')

    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8
    )

    renderer.update_scene(env.data, camera=cam_id)
    frame = renderer.render()
    writer.append_data(frame)

    for step in range(1, args.steps + 1):
        # Action sinusoïdale, chaque doigt avec un déphasage unique
        phase = np.linspace(0, np.pi, env.action_space.shape[0])
        action = np.sin(2 * np.pi * step / 40.0 + phase)
        env.data.ctrl[:] = action
        mj_step(env.model, env.data)
        viewer.sync()
        renderer.update_scene(env.data, camera=cam_id)
        frame = renderer.render()
        writer.append_data(frame)
        print(f"\rStep {step}/{args.steps}", end="", flush=True)
        if not viewer.is_running():
            print("\n⚠ Fenêtre fermée manuellement, arrêt de la simulation.")
            break

    writer.close()
    viewer.close()
    env.close()
    print(f"\n✅ Vidéo générée avec succès : {out_path.resolve()}")

if __name__ == "__main__":
    main()
