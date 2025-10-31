# Bug Report: scipy.stats.entropy KL Divergence Negative

**Target**: `scipy.stats.entropy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.stats.entropy` function can return negative KL divergence values when computing relative entropy with extremely small probabilities, violating the mathematical property that KL divergence must always be non-negative.

## Property-Based Test

```python
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
```

**Failing input**:
- `pk=[1.0, 1.6201705867724093e-138]`
- `qk=[1.0, 1.3310649736289993e-42]`

## Reproducing the Bug

```python
import numpy as np
import scipy.stats

pk = np.array([1.0, 1.62e-138])
qk = np.array([1.0, 1.33e-42])

kl = scipy.stats.entropy(pk, qk)
print(f"KL divergence: {kl}")
```

Output:
```
KL divergence: -3.577784931870767e-136
```

## Why This Is A Bug

KL divergence (Kullback-Leibler divergence) is defined as:

D_KL(P||Q) = Σ p_i * log(p_i / q_i)

By Gibbs' inequality, KL divergence is always non-negative: D_KL(P||Q) ≥ 0, with equality if and only if P = Q.

The bug occurs when computing KL divergence with extremely small probabilities (on the order of 1e-138). The computation involves:
- pk[1] = 1.62e-138 (extremely small probability)
- qk[1] = 1.33e-42 (small but much larger than pk[1])
- Contribution: 1.62e-138 * log(1.62e-138 / 1.33e-42) = 1.62e-138 * (-221) ≈ -3.6e-136

Due to floating-point underflow and the extreme magnitude differences, this negative contribution causes the total KL divergence to become negative, violating the fundamental mathematical property.

## Fix

The issue stems from computing with probabilities near the limits of floating-point precision. A fix could involve:

1. **Thresholding**: Filter out extremely small probabilities before computation
2. **Numerical stability**: Use log-space computations to avoid underflow
3. **Validation**: Clamp the final result to be non-negative with a warning

A simple fix would be to add a final check that clamps negative values to zero when they're due to numerical error:

```diff
--- a/scipy/stats/_stats_py.py
+++ b/scipy/stats/_stats_py.py
@@ -XXXX,X +XXXX,X @@ def entropy(...):
     ...
     S = xp.sum(pk * xp.log(pk / qk), axis=axis)
+    # Ensure non-negativity (KL divergence must be >= 0)
+    # Negative values close to zero are due to numerical precision
+    if qk is not None:
+        S = xp.where(xp.abs(S) < 1e-100, 0.0, S)
+        S = xp.maximum(S, 0.0)
     return S
```

However, a more robust solution would be to use numerically stable algorithms for computing KL divergence in log-space.