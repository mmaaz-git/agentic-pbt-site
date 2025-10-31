import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats
from hypothesis import given, strategies as st, assume, settings

@given(
    pk=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    qk=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50)
)
@settings(max_examples=500)
def test_entropy_kl_divergence_nonnegative(pk, qk):
    n = min(len(pk), len(qk))
    pk = np.array(pk[:n])
    qk = np.array(qk[:n])

    assume(pk.sum() > 1e-10)
    assume(qk.sum() > 1e-10)
    assume(np.all(pk > 0))
    assume(np.all(qk > 0))

    kl = scipy.stats.entropy(pk, qk)
    assert kl >= 0, f"KL divergence {kl} is negative"

if __name__ == "__main__":
    test_entropy_kl_divergence_nonnegative()