import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats

# Example from bug report
pk = np.array([1.0, 1.62e-138])
qk = np.array([1.0, 1.33e-42])

kl = scipy.stats.entropy(pk, qk)
print(f"KL divergence: {kl}")
print(f"Is negative: {kl < 0}")

# Additional test with the found example from hypothesis
pk2 = np.array([1.0, 1.8176053134562562e-246])
qk2 = np.array([1.0, 1.401298464324817e-45])

kl2 = scipy.stats.entropy(pk2, qk2)
print(f"\nAlternative example KL divergence: {kl2}")
print(f"Is negative: {kl2 < 0}")