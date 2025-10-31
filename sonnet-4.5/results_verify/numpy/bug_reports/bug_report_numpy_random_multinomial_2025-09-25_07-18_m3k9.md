# Bug Report: numpy.random.multinomial Silently Produces Incorrect Distribution

**Target**: `numpy.random.multinomial`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `numpy.random.multinomial` is called with probabilities that don't sum to 1, it silently accepts them and produces an incorrect distribution instead of either raising a ValueError or properly normalizing the probabilities.

## Property-Based Test

```python
import numpy as np
import numpy.random
from hypothesis import given, settings, strategies as st


@given(st.integers(min_value=100, max_value=10000))
@settings(max_examples=20)
def test_multinomial_unnormalized_probabilities(n):
    pvals = [0.5, 0.5, 0.5]

    result = numpy.random.multinomial(n, pvals)
    proportions = result / n

    expected_if_normalized = np.array([1/3, 1/3, 1/3])
    differences = np.abs(proportions - expected_if_normalized)
    max_diff = np.max(differences)

    assert max_diff <= 0.15, f"multinomial produces incorrect distribution: {proportions}"
```

**Failing input**: `n=100, pvals=[0.5, 0.5, 0.5]`

## Reproducing the Bug

```python
import numpy as np
import numpy.random

pvals = [0.5, 0.5, 0.5]
n = 10000

result = numpy.random.multinomial(n, pvals)
proportions = result / n

print(f"Input probabilities: {pvals} (sum={sum(pvals)})")
print(f"Result: {result}")
print(f"Proportions: {proportions}")

print(f"\nExpected (if normalized): ~[0.333, 0.333, 0.333]")
print(f"Actual: [{proportions[0]:.3f}, {proportions[1]:.3f}, {proportions[2]:.3f}]")
```

Output:
```
Input probabilities: [0.5, 0.5, 0.5] (sum=1.5)
Result: [5019 4981    0]
Proportions: [0.5019 0.4981 0.    ]

Expected (if normalized): ~[0.333, 0.333, 0.333]
Actual: [0.502, 0.498, 0.000]
```

## Why This Is A Bug

The documentation for `numpy.random.multinomial` explicitly states: "The probability inputs should be normalized" and "must sum to 1". However, when given `pvals=[0.5, 0.5, 0.5]` (which sums to 1.5), the function:

1. **Silently accepts** the invalid input instead of raising a ValueError
2. **Produces incorrect results**: consistently assigns ~50% to the first two categories and 0% to the third
3. **Violates the documented contract**: probabilities should sum to 1

If the probabilities were properly normalized, the third category should receive ~33% of samples (1/3 of the total), not 0%.

This is a high-severity bug because:
- It causes **silent data corruption** - wrong results without any warning
- Users may not realize their probabilities don't sum to 1
- The behavior is mathematically incorrect

## Fix

The function should validate that probabilities sum to 1 (within a small tolerance). Add validation in `multinomial`:

```diff
def multinomial(n, pvals, size=None):
+    pvals_array = np.asarray(pvals)
+    psum = pvals_array.sum()
+    if not np.isclose(psum, 1.0, rtol=1e-7):
+        raise ValueError(f"pvals must sum to 1 (got {psum})")
+
    # existing implementation
    ...
```

Alternatively, if auto-normalization is desired, the function should normalize properly:

```diff
def multinomial(n, pvals, size=None):
+    pvals_array = np.asarray(pvals)
+    pvals_normalized = pvals_array / pvals_array.sum()
+    # use pvals_normalized instead of pvals
    ...
```