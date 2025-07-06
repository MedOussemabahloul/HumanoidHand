print("Import de utils/reward_utils.py réussi.")

def distance_reward(x, y):
    r = -abs(x - y)
    print(f"Reward calculé (distance main-cube): {r}")
    return r
