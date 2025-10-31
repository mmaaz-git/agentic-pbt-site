# Bug Report: numpy.random.dirichlet Produces NaN/Inf Values with Small Alpha Parameters

**Target**: `numpy.random.dirichlet`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.random.dirichlet` function produces NaN and infinity values when given small positive alpha parameters (e.g., < 0.01), violating the fundamental mathematical property that Dirichlet samples must be valid probability vectors with all components positive and summing to 1.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume


@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=10))
def test_dirichlet_sums_to_one(alpha):
    assume(all(a > 0 for a in alpha))
    samples = np.random.dirichlet(alpha, size=100)
    sums = samples.sum(axis=1)
    assert np.allclose(sums, 1.0)


if __name__ == "__main__":
    test_dirichlet_sums_to_one()
```

<details>

<summary>
<strong>Failing input</strong>: <code>alpha=[0.001953125, 0.00390625]</code>
</summary>

```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 15, in <module>
    test_dirichlet_sums_to_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_dirichlet_sums_to_one
    min_size=2, max_size=10))
    ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 11, in test_dirichlet_sums_to_one
    assert np.allclose(sums, 1.0)
           ~~~~~~~~~~~^^^^^^^^^^^
AssertionError
Falsifying example: test_dirichlet_sums_to_one(
    alpha=[0.001953125, 0.00390625],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

np.random.seed(42)
alpha = [0.001953125, 0.00390625]
samples = np.random.dirichlet(alpha, size=1000)

print(f"Alpha values: {alpha}")
print(f"Total samples generated: {len(samples)}")
print(f"Samples with NaN: {np.isnan(samples).any(axis=1).sum()}")
print(f"Samples with inf: {np.isinf(samples).any(axis=1).sum()}")
print(f"Samples with -inf: {np.isneginf(samples).any(axis=1).sum()}")

# Show some invalid samples
invalid_mask = np.isnan(samples).any(axis=1) | np.isinf(samples).any(axis=1)
invalid_samples = samples[invalid_mask]

if len(invalid_samples) > 0:
    print(f"\nFirst 5 invalid samples:")
    for i, sample in enumerate(invalid_samples[:5]):
        print(f"  Sample {i+1}: {sample}")
        print(f"    Sum: {np.nansum(sample)}")
        print(f"    Has NaN: {np.isnan(sample).any()}, Has Inf: {np.isinf(sample).any()}")

# Check valid samples
valid_samples = samples[~invalid_mask]
if len(valid_samples) > 0:
    print(f"\nValid samples: {len(valid_samples)}")
    sums = valid_samples.sum(axis=1)
    print(f"  Min sum: {sums.min():.10f}")
    print(f"  Max sum: {sums.max():.10f}")
    print(f"  Mean sum: {sums.mean():.10f}")
    print(f"  All sums close to 1.0: {np.allclose(sums, 1.0)}")
```

<details>

<summary>
20 out of 1000 samples contain NaN or infinity values
</summary>

```
Alpha values: [0.001953125, 0.00390625]
Total samples generated: 1000
Samples with NaN: 18
Samples with inf: 2
Samples with -inf: 0

First 5 invalid samples:
  Sample 1: [nan inf]
    Sum: inf
    Has NaN: True, Has Inf: True
  Sample 2: [nan nan]
    Sum: 0.0
    Has NaN: True, Has Inf: False
  Sample 3: [nan nan]
    Sum: 0.0
    Has NaN: True, Has Inf: False
  Sample 4: [nan nan]
    Sum: 0.0
    Has NaN: True, Has Inf: False
  Sample 5: [nan inf]
    Sum: inf
    Has NaN: True, Has Inf: True

Valid samples: 982
  Min sum: 1.0000000000
  Max sum: 1.0000000000
  Mean sum: 1.0000000000
  All sums close to 1.0: True
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical definition and documented behavior of the Dirichlet distribution:

1. **Documentation states**: "The Dirichlet distribution is a distribution over vectors x that fulfil the conditions x_i>0 and ∑_{i=1}^k x_i = 1"
2. **Function accepts the input**: Alpha values > 0 are mathematically valid and accepted without error
3. **Output violates contract**: NaN and infinity are NOT > 0, and NaN/inf sums don't equal 1
4. **Silent data corruption**: No error or warning is raised when producing invalid samples
5. **Inconsistency**: The newer `np.random.Generator.dirichlet` handles these same inputs correctly

The bug occurs because the legacy implementation uses gamma distributions internally, which suffer from numerical underflow with very small shape parameters. When gamma(alpha) underflows to 0 for all components, the normalization step produces 0/0 = NaN or divides by near-zero values producing infinity.

## Relevant Context

- This is a known issue dating back to 2015 (GitHub issue [#5851](https://github.com/numpy/numpy/issues/5851))
- The new Generator API (introduced in NumPy 1.17) correctly handles these inputs using a numerically stable stick-breaking algorithm for small alpha values
- Testing shows the new API produces valid samples for the same problematic alpha values:
  ```python
  rng = np.random.default_rng(42)
  samples = rng.dirichlet([0.001953125, 0.00390625], size=1000)
  # Result: 0 NaN, 0 inf, all sums = 1.0
  ```
- Small alpha values (< 0.01) create highly sparse distributions, which are legitimate use cases in machine learning (e.g., sparse topic models, Dirichlet process priors)
- NumPy documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html
- Source implementation uses gamma sampling followed by normalization

## Proposed Fix

Since the new Generator API already solves this problem, the legacy API should either adopt the same approach or add input validation. The minimal fix would be to add validation that rejects problematic inputs:

```diff
--- a/numpy/random/mtrand.pyx
+++ b/numpy/random/mtrand.pyx
@@ -4826,6 +4826,10 @@ cdef class RandomState:
         if np.any(np.less_equal(alpha_arr, 0)):
             raise ValueError('All values in alpha must be greater than 0')
+
+        # Add numerical stability check
+        if np.any(np.less(alpha_arr, 1e-3)):
+            raise ValueError('Alpha values below 1e-3 may cause numerical instability. '
+                           'Use np.random.default_rng().dirichlet() for small alpha values.')

         shape = np.append(size, alpha_arr.shape[-1])
```

A better fix would be to backport the stick-breaking algorithm used in the Generator API for cases where `max(alpha) < 0.1`, ensuring backward compatibility while fixing the numerical instability.