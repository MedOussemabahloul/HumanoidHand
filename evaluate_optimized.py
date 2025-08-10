"""
🎥 SCRIPT D'ÉVALUATION ET VIDÉO
==============================

Script simple pour évaluer un modèle entraîné et créer une vidéo
"""

import numpy as np
from stable_baselines3 import TD3
from envs.optimized_grasp_env import OptimizedGraspEnv
import imageio
from PIL import Image

def evaluate_model(model_path: str, video_path: str = "evaluation.mp4"):
    """Évalue un modèle et crée une vidéo"""

    print(f"🔄 Chargement du modèle: {model_path}")
    model = TD3.load(model_path)

    print("🏗️ Création de l'environnement...")
    env = OptimizedGraspEnv(render_mode="rgb_array")

    print("🎥 Enregistrement de la vidéo...")
    obs, _ = env.reset()
    frames = []

    total_reward = 0
    max_contacts = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

        total_reward += reward
        max_contacts = max(max_contacts, info.get('contact_count', 0))

        # Capturer frame
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame.astype(np.uint8)))

        if step % 100 == 0:
            print(f"   Step {step}: reward={total_reward:.1f}, contacts={max_contacts}")

        if terminated:
            break

    # Sauvegarder vidéo
    imageio.mimsave(video_path, frames, fps=30)

    print(f"✅ Vidéo sauvegardée: {video_path}")
    print(f"📊 Reward total: {total_reward:.2f}")
    print(f"🤝 Contacts max: {max_contacts}")

    env.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "optimized_results/models/best_model"

    evaluate_model(model_path)