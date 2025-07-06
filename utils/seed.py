import numpy as np
import torch
import random

print("Import de utils/seed.py réussi.")

def set_seed(seed):
    print(f"Fixation du seed à {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
