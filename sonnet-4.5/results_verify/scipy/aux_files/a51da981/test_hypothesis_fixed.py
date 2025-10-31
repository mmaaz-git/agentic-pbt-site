import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats
from hypothesis import given, strategies as st, assume, settings

# Test the specific failing case first
print("Testing the specific failing case:")
samples = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0])
]

# Verify they have the same variance
for i, s in enumerate(samples):
    print(f"Sample {i+1}: variance = {np.var(s, ddof=1)}")

statistic, p = scipy.stats.bartlett(*samples)
print(f"Statistic: {statistic}")
print(f"P-value: {p}")
print(f"P-value is NaN: {np.isnan(p)}")

# Now run the hypothesis test
@given(
    samples=st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=3, max_size=50),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10)  # Reduced for testing
def test_bartlett_pvalue_bounds(samples):
    samples = [np.array(s) for s in samples]
    STD_THRESHOLD = 1e-6

    for s in samples:
        assume(len(s) >= 3)
        assume(np.std(s) > STD_THRESHOLD)

    statistic, p = scipy.stats.bartlett(*samples)
    assert 0 <= p <= 1, f"Bartlett p-value {p} outside [0, 1]"

print("\nRunning hypothesis test (10 examples)...")
test_bartlett_pvalue_bounds()