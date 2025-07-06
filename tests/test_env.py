from envs.humanoid_manip_env import HumanoidManipEnv
import numpy as np

def test_env_reset():
    print("=== Test : reset de l'environnement ===")
    env = HumanoidManipEnv()
    obs, info = env.reset()
    print("Observation après reset :", obs)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert np.allclose(obs, [0.0, 0.0]), "Reset n'a pas bien réinitialisé l'état"

def test_env_step_random_action():
    print("=== Test : step avec action aléatoire ===")
    env = HumanoidManipEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Nouvelle obs: {obs2}, reward: {reward}")
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated

def test_env_step_limits():
    print("=== Test : step avec action aux limites ===")
    env = HumanoidManipEnv()
    obs, info = env.reset()
    actions = [env.action_space.low, env.action_space.high]
    for a in actions:
        obs2, reward, terminated, truncated, info = env.step(a)
        print(f"Action limite: {a}, Obs: {obs2}, Reward: {reward}")
        assert obs2.shape == (2,)
        assert isinstance(reward, float)

def test_env_reward_signal():
    import numpy as np
    from envs.humanoid_manip_env import HumanoidManipEnv

    env = HumanoidManipEnv()
    env.reset()
    # main très loin du cube
    env.data.qpos[0] = -0.5
    env.data.qpos[1] = 0.5
    obs, reward, *_ = env.step([0.0])
    assert reward < 0

    # main et cube très proches
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    obs, reward, *_ = env.step([0.0])
    # On tolère les petites erreurs de MuJoCo
    assert np.isclose(reward, 0.0, atol=1e-2)

def test_env_rollout():
    print("=== Test : rollout sur 10 steps ===")
    env = HumanoidManipEnv()
    obs, info = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[{i}] action={action}, obs={obs}, reward={reward}")
        assert obs.shape == (2,)
        assert isinstance(reward, float)

# Pytest les exécutera tous automatiquement
if __name__ == "__main__":
    test_env_reset()
    test_env_step_random_action()
    test_env_step_limits()
    test_env_reward_signal()
    test_env_rollout()
