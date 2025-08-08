#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import imageio
import imageio_ffmpeg

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.ultra_robust_grasp_env import make_ultra_robust_grasp_env

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args():
    p = argparse.ArgumentParser(description="Générer une vidéo d'un policy entraîné (SB3)")
    p.add_argument("--model", type=Path, required=True, help="Chemin vers le modèle .zip SB3")
    p.add_argument("--vecnorm", type=Path, default=None, help="Chemin vecnormalize.pkl si existant")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("videos/rollout.mp4"))
    return p.parse_args()


def build_eval_env(vecnorm_path: Path | None):
    # Base env (non viewer, curriculum off pour stabilité)
    base_fn = lambda: make_ultra_robust_grasp_env(
        render_mode="rgb_array", enable_curriculum=False, enable_mujoco_viewer=False
    )
    venv = DummyVecEnv([base_fn])
    if vecnorm_path and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
    return venv


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"▶ Génération vidéo : steps={args.steps}, fps={args.fps}")
    print(f"  Modèle : {args.model}")
    if args.vecnorm and args.vecnorm.exists():
        print(f"  VecNormalize: {args.vecnorm}")
    print(f"  Vidéo sortie : {args.out}\n")

    # Charger env + modèle
    env = build_eval_env(args.vecnorm)
    model = SAC.load(str(args.model))

    # Préparer writer
    writer = imageio.get_writer(
        str(args.out), fps=args.fps, codec="libx264", quality=8
    )

    # Reset
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Rollout
    for step in range(1, args.steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        # Render depuis l'env interne
        try:
            # Extraire env interne Gym pour appeler render(mode='rgb_array')
            inner = env.venv.envs[0]
            while hasattr(inner, 'env'):
                inner = inner.env
            frame = inner.render(mode='rgb_array')
        except Exception:
            frame = None
        if frame is not None:
            writer.append_data(frame)

        if bool(np.any(dones)):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

        if step % 50 == 0:
            print(f"\rStep {step}/{args.steps}", end="", flush=True)

    writer.close()
    env.close()
    print(f"\n✅ Vidéo générée avec succès : {args.out.resolve()}")


if __name__ == "__main__":
    main()
