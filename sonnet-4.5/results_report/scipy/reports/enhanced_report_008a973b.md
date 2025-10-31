# Bug Report: scipy.stats.entropy Returns Negative KL Divergence Values

**Target**: `scipy.stats.entropy`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.stats.entropy` function returns negative Kullback-Leibler (KL) divergence values when computing relative entropy with extremely small probabilities, violating the fundamental mathematical property that KL divergence must always be non-negative.

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

if __name__ == "__main__":
    test_entropy_kl_divergence_nonnegative()
```

<details>

<summary>
**Failing input**: `pk=[2.0007560879758843e-213, 1.0], qk=[1.432375450419533e-60, 1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 27, in <module>
    test_entropy_kl_divergence_nonnegative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 9, in test_entropy_kl_divergence_nonnegative
    pk=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 24, in test_entropy_kl_divergence_nonnegative
    assert kl >= 0, f"KL divergence {kl} is negative"
           ^^^^^^^
AssertionError: KL divergence -7.041887703187054e-211 is negative
Falsifying example: test_entropy_kl_divergence_nonnegative(
    pk=[2.0007560879758843e-213, 1.0],
    qk=[1.432375450419533e-60, 1.0],
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
Output showing negative KL divergence
</summary>
```
KL divergence: -3.577784931870767e-136
KL divergence is negative: True
This violates Gibbs' inequality (KL divergence must be >= 0)
```
</details>

## Why This Is A Bug

The Kullback-Leibler (KL) divergence, also known as relative entropy, is mathematically defined as:

D_KL(P||Q) = Σ p_i * log(p_i / q_i)

By Gibbs' inequality, a fundamental theorem in information theory, KL divergence must satisfy D_KL(P||Q) ≥ 0, with equality if and only if P = Q almost everywhere. This non-negativity property is proven through Jensen's inequality and is universally accepted in mathematical literature.

The bug manifests when processing probability distributions with extremely small values (on the order of 1e-138 or smaller). The function computes individual terms using `scipy.special.rel_entr(pk, qk)` which calculates `pk * log(pk/qk)`. When pk is extremely small (e.g., 1.62e-138) and qk is relatively larger (e.g., 1.33e-42), the computation yields:

1. pk/qk ≈ 1.22e-96 (very small ratio)
2. log(pk/qk) ≈ -220.85 (large negative value)
3. pk * log(pk/qk) ≈ 1.62e-138 * (-220.85) ≈ -3.58e-136 (negative result)

This negative contribution, when summed with other terms, causes the total KL divergence to become negative. The issue stems from floating-point arithmetic limitations when dealing with denormalized numbers near the limits of double precision representation.

## Relevant Context

The scipy.stats.entropy function (located in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/stats/_entropy.py`) is widely used in scientific computing for:
- Information theory calculations
- Machine learning model evaluation
- Statistical analysis
- Cross-entropy loss computation

The function documentation states it computes "relative entropy D = sum(pk * log(pk / qk))" which is "also known as the Kullback-Leibler divergence" (lines 37-39 of _entropy.py). The implementation normalizes inputs and uses `scipy.special.rel_entr` for the actual computation (line 155).

According to scipy.special.rel_entr documentation, it returns `x * log(x/y)` when `x > 0, y > 0`, which should theoretically be non-negative when summed over a probability distribution. However, numerical precision issues with extremely small values cause violations of this mathematical invariant.

Documentation references:
- scipy.stats.entropy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
- Gibbs' inequality: https://en.wikipedia.org/wiki/Gibbs%27_inequality

## Proposed Fix

```diff
--- a/scipy/stats/_entropy.py
+++ b/scipy/stats/_entropy.py
@@ -154,7 +154,13 @@ def entropy(pk: np.typing.ArrayLike,
         qk = 1.0*qk / xp.sum(qk, **sum_kwargs)  # type: ignore[operator, call-overload]
         vec = special.rel_entr(pk, qk)
     S = xp.sum(vec, axis=axis)
+
+    # Ensure KL divergence is non-negative (required by Gibbs' inequality)
+    # Small negative values can occur due to floating-point precision errors
+    if qk is not None:
+        S = xp.where(xp.logical_and(S < 0, xp.abs(S) < 1e-100), 0.0, S)
+        S = xp.maximum(S, 0.0)
+
     if base is not None:
         S /= math.log(base)
     return S
```