import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats

# Test case from the bug report
pk = np.array([1.0, 1.62e-138])
qk = np.array([1.0, 1.33e-42])

kl = scipy.stats.entropy(pk, qk)
print(f"KL divergence: {kl}")
print(f"KL divergence is negative: {kl < 0}")
print(f"This violates Gibbs' inequality (KL divergence must be >= 0)")