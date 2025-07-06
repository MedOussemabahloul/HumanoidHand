import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from stable_baselines3 import SAC
import yaml
import os

from envs.humanoid_manip_env import HumanoidManipEnv

def main():
    # Load config
    with open("configs/sac_default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Register l'environnement custom
    gym.register(
        id="HumanoidManip-v0",
        entry_point="envs.humanoid_manip_env:HumanoidManipEnv"
    )

    env = gym.make(config["env_name"])
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=config["total_timesteps"])
    os.makedirs("./models/", exist_ok=True)
    model.save("./models/sac_humanoid_manip")

if __name__ == "__main__":
    main()
