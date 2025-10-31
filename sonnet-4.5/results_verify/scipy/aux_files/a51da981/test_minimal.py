import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats

print("Testing the minimal reproduction case from bug report:")
samples = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0])
]

statistic, p = scipy.stats.bartlett(*samples)
print(f"Statistic: {statistic}")
print(f"P-value: {p}")