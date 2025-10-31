# Bug Report: pandas.plotting.bootstrap_plot Uses Sampling Without Replacement

**Target**: `pandas.plotting._matplotlib.bootstrap_plot`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `bootstrap_plot` function uses `random.sample()` which samples WITHOUT replacement, but bootstrapping by definition requires sampling WITH replacement. This produces statistically incorrect bootstrap estimates.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from collections import Counter

@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=5, max_size=20))
def test_bootstrap_should_sample_with_replacement(data):
    series = pd.Series(data)
    size = len(data)

    import random
    series_data = list(series.values)

    found_duplicate = False
    for _ in range(100):
        sample = random.sample(series_data, size)
        if len(sample) != len(set(sample)):
            found_duplicate = True
            break

    assert found_duplicate, "Bootstrap must sample WITH replacement"
```

**Failing input**: Any series - `random.sample()` never produces duplicates

## Reproducing the Bug

```python
import pandas as pd
import random

series = pd.Series([1, 2, 3])
random.seed(42)

data = list(series.values)
size = 3
samples = 1000

samplings = [random.sample(data, size) for _ in range(samples)]

has_duplicates = any(len(s) != len(set(s)) for s in samplings)

print(f"Any sample has duplicates: {has_duplicates}")
print(f"Expected for bootstrap (WITH replacement): True")
print(f"Actual result: {has_duplicates}")
```

Output:
```
Any sample has duplicates: False
Expected for bootstrap (WITH replacement): True
Actual result: False
```

## Why This Is A Bug

The function's docstring explicitly states:

> "The bootstrap plot is used to estimate the uncertainty of a statistic by relying on random sampling **with replacement**"

However, the implementation uses `random.sample(data, size)` which samples **without replacement**. This violates:

1. The documented behavior
2. The fundamental definition of bootstrapping
3. Statistical validity - bootstrap estimates will be incorrect

Bootstrap requires sampling with replacement to work correctly. Using `random.sample()` means each bootstrap sample is just a permutation of unique elements, not a proper bootstrap sample.

## Fix

```diff
--- a/pandas/plotting/_matplotlib/bootstrap.py
+++ b/pandas/plotting/_matplotlib/bootstrap.py
@@ -15,7 +15,7 @@ def bootstrap_plot(
     import matplotlib.pyplot as plt

     data = list(series.values)
-    samplings = [random.sample(data, size) for _ in range(samples)]
+    samplings = [random.choices(data, k=size) for _ in range(samples)]

     means = np.array([np.mean(sampling) for sampling in samplings])
     medians = np.array([np.median(sampling) for sampling in samplings])
```

The fix changes `random.sample(data, size)` to `random.choices(data, k=size)` which correctly samples with replacement.