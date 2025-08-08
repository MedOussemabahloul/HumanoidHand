#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys

# Dossiers par défaut
RESULTS = Path("robust_sac_results")
MODELS = RESULTS / "models"
VIDEOS = RESULTS / "videos"


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode


def latest_model() -> Path | None:
    if not MODELS.exists():
        return None
    zips = sorted(MODELS.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def maybe_vecnorm() -> Path | None:
    p = MODELS / "vecnormalize.pkl"
    return p if p.exists() else None


def main():
    # 1) Lancer l'entraînement SAC robuste
    run([sys.executable, str(Path(__file__).parent.parent / "robust_training_sac.py")])

    # 2) Chercher le dernier modèle et générer une vidéo
    model = latest_model()
    if model is None:
        print("Aucun modèle trouvé après l'entraînement.")
        return
    vecnorm = maybe_vecnorm()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = VIDEOS / "auto_demo.mp4"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "rollout_video.py"),
        "--model", str(model),
        "--steps", "800",
        "--fps", "30",
        "--out", str(out),
    ]
    if vecnorm:
        cmd += ["--vecnorm", str(vecnorm)]
    run(cmd)
    print(f"✅ Démo vidéo: {out}")


if __name__ == "__main__":
    main()
