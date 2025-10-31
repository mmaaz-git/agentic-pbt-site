from scipy.spatial.transform import Rotation
import numpy as np

q = np.array([0.0, 0.0, 0.0, 1.0])
r = Rotation.from_quat(q)

print("Attempting to call Rotation.mean([r]) with a single rotation in a list...")
r_mean = Rotation.mean([r])
print("Mean computed:", r_mean.as_quat())