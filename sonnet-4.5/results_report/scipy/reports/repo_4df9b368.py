import numpy as np
from scipy.cluster.vq import whiten

# Create an array with 7 identical values
obs = np.array([[5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075]])

# Check the standard deviation
std_val = np.std(obs, axis=0)[0]
print(f"std_val: {std_val}")
print(f"std_val == 0: {std_val == 0}")

# Apply whiten function
whitened = whiten(obs)
print(f"Original value: {obs[0, 0]}")
print(f"Whitened value: {whitened[0, 0]}")

# Verify all values are identical
print(f"All values identical: {np.all(obs == obs[0, 0])}")

# Mathematical verification
print(f"Division result: {obs[0, 0] / std_val}")