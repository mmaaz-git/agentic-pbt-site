# Bug Report: pandas.plotting.andrews_curves Produces Incorrect Curves for Odd-Dimensioned Data

**Target**: `pandas.plotting.andrews_curves`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `andrews_curves` function produces mathematically incorrect curves when the input DataFrame has an even number of total columns (odd number of feature columns after excluding the class column). This is caused by misuse of `np.resize` which repeats coefficients instead of padding with zeros, violating the Andrews curves mathematical formula.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_andrews_curve_expected(amplitudes, t):
    x1 = amplitudes[0]
    result = x1 / np.sqrt(2.0)

    coeffs = amplitudes[1:]
    n_pairs = len(coeffs) // 2
    has_odd = len(coeffs) % 2 == 1

    for i in range(n_pairs):
        sin_coeff = coeffs[2*i]
        cos_coeff = coeffs[2*i + 1]
        harmonic = i + 1
        result += sin_coeff * np.sin(harmonic * t) + cos_coeff * np.cos(harmonic * t)

    if has_odd:
        last_coeff = coeffs[-1]
        harmonic = n_pairs + 1
        result += last_coeff * np.sin(harmonic * t)

    return result


@given(
    n_features=st.integers(min_value=4, max_value=10).filter(lambda x: x % 2 == 0)
)
@settings(max_examples=20)
def test_andrews_curves_odd_coefficients_bug(n_features):
    data = {f'f{i}': [float(i+1)] for i in range(n_features)}
    data['class'] = ['A']
    df = pd.DataFrame(data)

    t_test = 0.0
    amplitudes = np.array([float(i+1) for i in range(n_features)])
    expected = compute_andrews_curve_expected(amplitudes, t_test)

    fig, ax = plt.subplots()
    pd.plotting.andrews_curves(df, 'class', samples=100, ax=ax)
    lines = ax.get_lines()

    for line in lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        t_idx = np.argmin(np.abs(xdata))
        actual = ydata[t_idx]

        assert np.isclose(actual, expected, rtol=1e-5), \
            f"Features {n_features}: Expected {expected}, got {actual}"

    plt.close(fig)
```

**Failing input**: `n_features=4` (any even number ≥ 4)

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'x': [1.0],
    'y': [2.0],
    'z': [3.0],
    'w': [4.0],
    'class': ['A']
})

t = 0.0
x1, x2, x3, x4 = 1.0, 2.0, 3.0, 4.0

expected = x1/np.sqrt(2.0) + x2*np.sin(t) + x3*np.cos(t) + x4*np.sin(2*t)
print(f"Expected f(0) = {x1}/√2 + {x2}·sin(0) + {x3}·cos(0) + {x4}·sin(0) = {expected}")

ax = pd.plotting.andrews_curves(df, 'class', samples=100)
lines = ax.get_lines()
xdata = lines[0].get_xdata()
ydata = lines[0].get_ydata()
t_idx = np.argmin(np.abs(xdata))
actual = ydata[t_idx]

print(f"Actual f(≈0) = {actual}")
print(f"Difference: {abs(actual - expected)}")
```

Output:
```
Expected f(0) = 1.0/√2 + 2.0·sin(0) + 3.0·cos(0) + 4.0·sin(0) = 3.7071067811865475
Actual f(≈0) = 5.707106781186548
Difference: 2.0
```

## Why This Is A Bug

Andrews curves are defined by the mathematical formula:

```
f(t) = x₁/√2 + x₂·sin(t) + x₃·cos(t) + x₄·sin(2t) + x₅·cos(2t) + ...
```

When there's an odd number of coefficients after the first term, the last coefficient should only be used with `sin(kt)`, with no corresponding `cos(kt)` term (implicitly paired with 0).

However, the implementation uses `np.resize` which **repeats elements cyclically** when the target size is larger than the source. For 4 features:
- Coefficients after removing x₁: `[x₂, x₃, x₄]` (3 elements)
- Resized to (2, 2): `[[x₂, x₃], [x₄, x₂]]` (4 elements - **x₂ is incorrectly repeated**)
- This adds a spurious term: `x₂ · cos(2t)` to the formula

The bug is in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py`:

```python
coeffs = np.delete(np.copy(amplitudes), 0)
coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))
```

## Fix

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -409,7 +409,12 @@ def andrews_curves(
             # Take the rest of the coefficients and resize them
             # appropriately. Take a copy of amplitudes as otherwise numpy
             # deletes the element from amplitudes itself.
             coeffs = np.delete(np.copy(amplitudes), 0)
-            coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))
+            n_pairs = (coeffs.size + 1) // 2
+            # Pad with zeros instead of using resize (which repeats values)
+            if coeffs.size % 2 == 1:
+                coeffs = np.append(coeffs, 0)
+            coeffs = coeffs.reshape((n_pairs, 2))

             # Generate the harmonics and arguments for the sin and cos
             # functions.
```