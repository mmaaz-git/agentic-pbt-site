import scipy.special as sp
import numpy as np

# Test case from bug report
delta = 1e-200
r = 1.0

result = sp.pseudo_huber(delta, r)
print(f"sp.pseudo_huber({delta}, {r}) = {result}")
print(f"Is result finite? {np.isfinite(result)}")
print(f"Is result NaN? {np.isnan(result)}")

# Test with additional small delta values to confirm the pattern
test_deltas = [1e-100, 1e-150, 1e-190, 1e-200, 1e-250, 1e-300]
for d in test_deltas:
    res = sp.pseudo_huber(d, 1.0)
    print(f"delta={d}: result={res}, is_finite={np.isfinite(res)}")

# Test the specific failing input from Hypothesis
delta_hyp = 2.3581411596114265e-203
r_hyp = 1.0
result_hyp = sp.pseudo_huber(delta_hyp, r_hyp)
print(f"\nHypothesis failing input: sp.pseudo_huber({delta_hyp}, {r_hyp}) = {result_hyp}")
print(f"Is result finite? {np.isfinite(result_hyp)}")