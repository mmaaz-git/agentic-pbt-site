import numpy as np
from scipy.cluster.vq import whiten

obs = np.array([[5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075]])

std_val = np.std(obs, axis=0)[0]
print(f"std_val: {std_val}")
print(f"std_val == 0: {std_val == 0}")

whitened = whiten(obs)
print(f"Original value: {obs[0, 0]}")
print(f"Whitened value: {whitened[0, 0]}")

# Additional verification
print(f"\nDivision result: {obs[0, 0] / std_val}")
print(f"Expected (if treated as zero std): {obs[0, 0]}")