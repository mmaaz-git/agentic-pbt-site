import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats
from hypothesis import given, strategies as st, assume, settings

@given(
    samples=st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=3, max_size=50),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=300)
def test_bartlett_pvalue_bounds(samples):
    samples = [np.array(s) for s in samples]
    STD_THRESHOLD = 1e-6

    for s in samples:
        assume(len(s) >= 3)
        assume(np.std(s) > STD_THRESHOLD)

    statistic, p = scipy.stats.bartlett(*samples)
    print(f"Testing samples with variances: {[np.var(s) for s in samples]}")
    print(f"Statistic: {statistic}, P-value: {p}")
    assert 0 <= p <= 1, f"Bartlett p-value {p} outside [0, 1]"

# Also test the specific failing case
print("\nTesting the specific failing case:")
samples = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0])
]
test_bartlett_pvalue_bounds(samples)