#!/usr/bin/env python3
"""
train_master_grasp.py

Robust SAC training runner with:
- CurriculumGraspEnv from your project
- Live MuJoCo viewer during training (passive sync)
- Reliable per-episode MP4 recording with codec fallbacks
- Minimal, clear, and resilient behavior

New file; does not overwrite existing ones.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Optional

import numpy as np

# Add envs and wrapper to sys.path
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ENV_DIR = "/home/oussema/Documents/project/envs"
if PROJECT_ENV_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ENV_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Safe imports
try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
except Exception as e:
    print(f"❌ Erreur import CurriculumGraspEnv: {e}")
    sys.exit(1)

try:
    from env_master_grasp_wrapper import LiveRenderVideoEnv
except Exception as e:
    print(f"❌ Erreur import wrapper: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
except Exception as e:
    print(f"❌ Erreur Stable-Baselines3: {e}")
    print("💡 pip install stable-baselines3 torch gymnasium")
    sys.exit(1)


def set_seeds(seed: int):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)


class ProgressGuardCallback(BaseCallback):
    """Monitor training progress to catch stagnation and instability.
    - Computes moving average of episode rewards
    - Logs curriculum info if available
    - Adjusts learning rate slightly if prolonged stagnation
    """
    def __init__(self, check_freq: int = 1000, window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window = window
        self.episode_rewards = []
        self.best_avg = -np.inf
        self.last_improvement_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if len(dones) and dones[0]:
            # Read episode reward from SB3 info if present
            ep = infos[0].get("episode", None) if len(infos) else None
            if ep is not None and "r" in ep:
                self.episode_rewards.append(ep["r"])
                if len(self.episode_rewards) >= self.window:
                    avg = float(np.mean(self.episode_rewards[-self.window:]))
                    if avg > self.best_avg + 1e-6:
                        self.best_avg = avg
                        self.last_improvement_step = self.num_timesteps
                    # Stagnation: no improvement for 50k steps -> mild LR tweak
                    if self.num_timesteps - self.last_improvement_step > 50000:
                        try:
                            old_lr = float(self.model.learning_rate)
                            new_lr = max(old_lr * 0.8, 1e-5)
                            self.model.learning_rate = new_lr
                            if self.verbose:
                                print(f"⚙️  Stagnation détectée, LR: {old_lr:.2e} → {new_lr:.2e}")
                            self.last_improvement_step = self.num_timesteps
                        except Exception:
                            pass
            # Curriculum metrics
            try:
                env0 = self.training_env.envs[0]
                if hasattr(env0, "get_curriculum_info"):
                    cur = env0.get_curriculum_info()
                    self.logger.record("curriculum/level", cur.get("current_level", 0))
                    self.logger.record("curriculum/phase", env0._get_phase_name())
            except Exception:
                pass
        return True


def run_training(total_timesteps: int, live_view: bool, record_video: bool, seed: int) -> Optional[str]:
    set_seeds(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"/home/oussema/Documents/project/master_results_{timestamp}"
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    videos_dir = os.path.join(base_dir, "videos")
    for d in (base_dir, models_dir, logs_dir, videos_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    print("🚀 LANCEMENT TRAIN MASTER GRASP")
    print(f"📁 Résultats: {base_dir}")

    # Create env with RGB rendering for video frames
    env = CurriculumGraspEnv(render_mode="rgb_array")
    env_wrapped = LiveRenderVideoEnv(
        env,
        live_view=live_view,
        record_video=record_video,
        video_dir=videos_dir,
        video_name_prefix="train",
        frame_size=(640, 480),
    )

    # Build SAC model with stable defaults
    model = SAC(
        "MlpPolicy",
        env_wrapped,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=logs_dir,
        device="auto",
        seed=seed,
    )
    model.set_logger(configure(logs_dir, ["stdout", "csv", "tensorboard"]))

    # Train with progress guard
    cb = ProgressGuardCallback(check_freq=1000, window=20, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb, log_interval=10)
    t_elapsed = time.time() - t0
    print(f"✅ Entraînement terminé en {t_elapsed:.1f}s")

    # Save final model
    model_path = os.path.join(models_dir, "sac_master_final.zip")
    model.save(model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")

    # Simple deterministic demo run to confirm video writing
    last_video_path: Optional[str] = None
    try:
        obs, info = env_wrapped.reset()
        ep_reward = 0.0
        for step in range(600):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_wrapped.step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                break
        print(f"🎬 Démo terminée, reward={ep_reward:.2f}")
        last_video_path = env_wrapped.current_video_path
    except Exception as e:
        print(f"⚠️  Démo non exécutée: {e}")

    # Persist a tiny training summary
    try:
        with open(os.path.join(base_dir, "summary.json"), "w") as f:
            json.dump({
                "total_timesteps": int(total_timesteps),
                "train_time_sec": float(t_elapsed),
                "model_path": model_path,
                "videos_dir": videos_dir,
                "last_video": last_video_path,
                "seed": seed,
            }, f, indent=2)
    except Exception:
        pass

    # Copy last video to a convenient demo path
    if last_video_path and os.path.exists(last_video_path):
        try:
            demo_path = os.path.join(base_dir, "demo_last_episode.mp4")
            if os.path.abspath(last_video_path) != os.path.abspath(demo_path):
                import shutil
                shutil.copy2(last_video_path, demo_path)
                print(f"📹 Copie démo: {demo_path}")
        except Exception:
            pass

    # Clean close
    try:
        env_wrapped.close()
    except Exception:
        pass

    return base_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Master Grasp Training (SAC)")
    p.add_argument("--timesteps", type=int, default=150000, help="Total timesteps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-view", action="store_true", help="Disable live MuJoCo viewer")
    p.add_argument("--no-video", action="store_true", help="Disable episode video recording")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = run_training(
        total_timesteps=args.timesteps,
        live_view=not args.no_view,
        record_video=not args.no_video,
        seed=args.seed,
    )
    print(f"🏁 Terminé. Résultats: {base_dir}")


if __name__ == "__main__":
    main()