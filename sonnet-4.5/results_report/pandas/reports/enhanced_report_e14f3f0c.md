# Bug Report: pandas.plotting.bootstrap_plot Uses Sampling Without Replacement

**Target**: `pandas.plotting._matplotlib.misc.bootstrap_plot`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `bootstrap_plot` function uses `random.sample()` which samples WITHOUT replacement, but bootstrapping by definition requires sampling WITH replacement, producing statistically incorrect bootstrap estimates.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import random
import numpy as np

@settings(max_examples=100)
@given(st.lists(st.integers(min_value=0, max_value=100), min_size=5, max_size=20))
def test_bootstrap_should_sample_with_replacement(data):
    """
    Test that bootstrap sampling produces duplicates (sampling WITH replacement).

    The bootstrap_plot function claims to use "random sampling with replacement",
    but it actually uses random.sample() which samples WITHOUT replacement.
    This test verifies that the current implementation never produces duplicates.
    """
    series = pd.Series(data)
    size = len(data)

    # This is what bootstrap_plot currently does
    series_data = list(series.values)

    # Try to find a duplicate in bootstrap samples
    found_duplicate = False
    for _ in range(100):
        sample = random.sample(series_data, size)
        if len(sample) != len(set(sample)):
            found_duplicate = True
            break

    # This assertion SHOULD pass for proper bootstrap (WITH replacement)
    # But it FAILS because random.sample() samples WITHOUT replacement
    assert found_duplicate, (
        f"Bootstrap must sample WITH replacement, but no duplicates found in 100 samples. "
        f"Data: {data}"
    )

if __name__ == "__main__":
    # Run the test
    test_bootstrap_should_sample_with_replacement()
    print("Test completed - check for failures above")
```

<details>

<summary>
**Failing input**: `[0, 1, 2, 3, 4]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 39, in <module>
    test_bootstrap_should_sample_with_replacement()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 7, in test_bootstrap_should_sample_with_replacement
    @given(st.lists(st.integers(min_value=0, max_value=100), min_size=5, max_size=20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 32, in test_bootstrap_should_sample_with_replacement
    assert found_duplicate, (
           ^^^^^^^^^^^^^^^
AssertionError: Bootstrap must sample WITH replacement, but no duplicates found in 100 samples. Data: [0, 1, 2, 3, 4]
Falsifying example: test_bootstrap_should_sample_with_replacement(
    data=[0, 1, 2, 3, 4],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import random

# Create a simple series for demonstration
series = pd.Series([1, 2, 3, 4, 5])
print(f"Original series: {list(series.values)}")
print()

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Demonstrate the current behavior using random.sample (WITHOUT replacement)
data = list(series.values)
size = len(data)
samples = 1000

print("Testing current implementation (random.sample - WITHOUT replacement):")
print("-" * 60)

# Generate bootstrap samples using random.sample (current implementation)
samplings_without = [random.sample(data, size) for _ in range(samples)]

# Check if any sample has duplicates
has_duplicates_without = any(len(s) != len(set(s)) for s in samplings_without)
print(f"Any sample has duplicates: {has_duplicates_without}")

# Calculate variance of means
means_without = [np.mean(s) for s in samplings_without]
variance_without = np.var(means_without)
print(f"Variance of bootstrap means: {variance_without:.6f}")

# Show first 5 samples to illustrate they are just permutations
print(f"\nFirst 5 samples (notice they're just permutations):")
for i, s in enumerate(samplings_without[:5]):
    print(f"  Sample {i+1}: {s}")

print()
print("Testing correct implementation (random.choices - WITH replacement):")
print("-" * 60)

# Reset seed for comparison
random.seed(42)

# Generate bootstrap samples using random.choices (correct implementation)
samplings_with = [random.choices(data, k=size) for _ in range(samples)]

# Check if samples have duplicates
has_duplicates_with = any(len(s) != len(set(s)) for s in samplings_with)
duplicate_count = sum(1 for s in samplings_with if len(s) != len(set(s)))
print(f"Any sample has duplicates: {has_duplicates_with}")
print(f"Number of samples with duplicates: {duplicate_count}/{samples} ({100*duplicate_count/samples:.1f}%)")

# Calculate variance of means
means_with = [np.mean(s) for s in samplings_with]
variance_with = np.var(means_with)
print(f"Variance of bootstrap means: {variance_with:.6f}")

# Show first 5 samples to illustrate they have duplicates
print(f"\nFirst 5 samples (notice the duplicates):")
for i, s in enumerate(samplings_with[:5]):
    print(f"  Sample {i+1}: {s} (unique elements: {len(set(s))})")

print()
print("Summary:")
print("=" * 60)
print(f"Bootstrap sampling MUST be done WITH replacement")
print(f"Current implementation (random.sample): NEVER produces duplicates")
print(f"Correct implementation (random.choices): produces duplicates ~{100*duplicate_count/samples:.0f}% of the time")
print(f"\nThis bug makes the bootstrap_plot function statistically incorrect!")
```

<details>

<summary>
Bootstrap variance is 0 with current implementation
</summary>
```
Original series: [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]

Testing current implementation (random.sample - WITHOUT replacement):
------------------------------------------------------------
Any sample has duplicates: False
Variance of bootstrap means: 0.000000

First 5 samples (notice they're just permutations):
  Sample 1: [np.int64(1), np.int64(5), np.int64(3), np.int64(2), np.int64(4)]
  Sample 2: [np.int64(2), np.int64(5), np.int64(3), np.int64(1), np.int64(4)]
  Sample 3: [np.int64(5), np.int64(4), np.int64(1), np.int64(3), np.int64(2)]
  Sample 4: [np.int64(2), np.int64(5), np.int64(3), np.int64(1), np.int64(4)]
  Sample 5: [np.int64(5), np.int64(4), np.int64(1), np.int64(2), np.int64(3)]

Testing correct implementation (random.choices - WITH replacement):
------------------------------------------------------------
Any sample has duplicates: True
Number of samples with duplicates: 956/1000 (95.6%)
Variance of bootstrap means: 0.368435

First 5 samples (notice the duplicates):
  Sample 1: [np.int64(4), np.int64(1), np.int64(2), np.int64(2), np.int64(4)] (unique elements: 3)
  Sample 2: [np.int64(4), np.int64(5), np.int64(1), np.int64(3), np.int64(1)] (unique elements: 4)
  Sample 3: [np.int64(2), np.int64(3), np.int64(1), np.int64(1), np.int64(4)] (unique elements: 4)
  Sample 4: [np.int64(3), np.int64(2), np.int64(3), np.int64(5), np.int64(1)] (unique elements: 4)
  Sample 5: [np.int64(5), np.int64(4), np.int64(2), np.int64(1), np.int64(5)] (unique elements: 4)

Summary:
============================================================
Bootstrap sampling MUST be done WITH replacement
Current implementation (random.sample): NEVER produces duplicates
Correct implementation (random.choices): produces duplicates ~96% of the time

This bug makes the bootstrap_plot function statistically incorrect!
```
</details>

## Why This Is A Bug

The `bootstrap_plot` function's docstring explicitly states it uses "random sampling **with replacement**" (line 402 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py`). However, the implementation at line 303 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py` uses:

```python
samplings = [random.sample(data, size) for _ in range(samples)]
```

The `random.sample()` function samples **WITHOUT replacement**, meaning each element can only appear once in each sample. This fundamentally violates the bootstrap methodology, which requires sampling WITH replacement to correctly estimate the sampling distribution of statistics.

The consequences are severe:
1. **Zero variance estimation**: Since every sample is just a permutation of the original data, they all have the same mean, resulting in zero variance
2. **Incorrect confidence intervals**: The function cannot properly estimate uncertainty
3. **Misleading visualizations**: The plots show no variability in the statistics when there should be substantial uncertainty
4. **Silent failure**: Users get results that look plausible but are statistically meaningless

## Relevant Context

Bootstrap sampling, as defined by Bradley Efron (1979), requires sampling WITH replacement. This allows the same observation to appear multiple times in a bootstrap sample, which is essential for correctly estimating the sampling distribution. The Wikipedia article referenced in the docstring (https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) confirms this requirement.

For a dataset of size n sampled with replacement:
- Probability that any sample has duplicates: 1 - n!/n^n (approximately 96% for n=5)
- With the current bug: Probability of duplicates is exactly 0%

The function is located at:
- Main interface: `pandas/plotting/_misc.py:391`
- Implementation: `pandas/plotting/_matplotlib/misc.py:291`

## Proposed Fix

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -300,7 +300,7 @@ def bootstrap_plot(
     # TODO: is the failure mentioned below still relevant?
     # random.sample(ndarray, int) fails on python 3.3, sigh
     data = list(series.values)
-    samplings = [random.sample(data, size) for _ in range(samples)]
+    samplings = [random.choices(data, k=size) for _ in range(samples)]

     means = np.array([np.mean(sampling) for sampling in samplings])
     medians = np.array([np.median(sampling) for sampling in samplings])
```