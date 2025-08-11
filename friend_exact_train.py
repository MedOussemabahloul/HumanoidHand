import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from friend_exact_env import CustomRobotEnv

# Create environment exactly like your friend
env = CustomRobotEnv()
env = Monitor(env)

print("Actuator count:", env.model.nu)
print("Right actuator IDs:", env.right_actuator_ids)

# TD3 with exact same config as friend
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space.shape[0]), 
    sigma=0.1 * np.ones(env.action_space.shape[0])
)

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    action_noise=action_noise,
    verbose=1
)

print("🚀 Training started - exactly like your friend!")
model.learn(total_timesteps=50000)

print("💾 Saving model...")
model.save("friend_exact_model")

print("🎉 Done! Model saved as 'friend_exact_model'")
print("🎬 Now you can generate video!"