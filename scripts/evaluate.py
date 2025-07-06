import gymnasium as gym
from stable_baselines3 import SAC

from envs.humanoid_manip_env import HumanoidManipEnv
from utils.logger import log

def main():
    log("Enregistrement de l'environnement HumanoidManip-v0")
    gym.register(
        id="HumanoidManip-v0",
        entry_point="envs.humanoid_manip_env:HumanoidManipEnv"
    )
    env = gym.make("HumanoidManip-v0", render_mode="human")
    log("Chargement du modèle SAC")
    model = SAC.load("./models/sac_humanoid_manip", env=env)
    obs, info = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
