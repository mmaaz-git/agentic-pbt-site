import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
from scipy import special
import scipy.stats

# Example from bug report
pk = np.array([1.0, 1.62e-138])
qk = np.array([1.0, 1.33e-42])

print("Input values:")
print(f"pk = {pk}")
print(f"qk = {qk}")
print(f"pk[0] = {pk[0]:.20e}, pk[1] = {pk[1]:.20e}")
print(f"qk[0] = {qk[0]:.20e}, qk[1] = {qk[1]:.20e}")

# Normalize (as scipy.stats.entropy does)
pk_norm = pk / pk.sum()
qk_norm = qk / qk.sum()

print(f"\nNormalized values:")
print(f"pk_norm = {pk_norm}")
print(f"qk_norm = {qk_norm}")
print(f"pk_norm sum = {pk_norm.sum()}")
print(f"qk_norm sum = {qk_norm.sum()}")

# Compute rel_entr for each element
rel_entr_vals = special.rel_entr(pk_norm, qk_norm)
print(f"\nrel_entr values:")
print(f"rel_entr_vals = {rel_entr_vals}")
print(f"rel_entr_vals[0] = {rel_entr_vals[0]:.20e}")
print(f"rel_entr_vals[1] = {rel_entr_vals[1]:.20e}")

# Sum to get KL divergence
kl = np.sum(rel_entr_vals)
print(f"\nKL divergence (sum): {kl:.20e}")

# Manual calculation
print(f"\nManual calculation:")
for i in range(len(pk_norm)):
    if pk_norm[i] > 0 and qk_norm[i] > 0:
        contrib = pk_norm[i] * np.log(pk_norm[i] / qk_norm[i])
        print(f"i={i}: pk_norm[i]={pk_norm[i]:.10e}, qk_norm[i]={qk_norm[i]:.10e}")
        print(f"      pk_norm[i]/qk_norm[i] = {pk_norm[i]/qk_norm[i]:.10e}")
        print(f"      log(pk_norm[i]/qk_norm[i]) = {np.log(pk_norm[i]/qk_norm[i]):.10e}")
        print(f"      contribution = {contrib:.20e}")

# Use scipy.stats.entropy
kl_scipy = scipy.stats.entropy(pk, qk)
print(f"\nUsing scipy.stats.entropy: {kl_scipy:.20e}")

# Check the mathematical constraint
print(f"\nIs KL divergence negative? {kl_scipy < 0}")
print(f"This violates the mathematical property that KL divergence >= 0")