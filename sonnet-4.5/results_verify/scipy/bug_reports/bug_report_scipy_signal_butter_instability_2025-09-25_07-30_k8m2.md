# Bug Report: scipy.signal Butterworth Filter Instability at Low Cutoff Frequencies

**Target**: `scipy.signal.butter` and `scipy.signal.cheby1`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Butterworth and Chebyshev Type I digital filters designed with very low cutoff frequencies produce unstable filters with poles outside the unit circle, violating the fundamental stability requirement for digital filters.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal


@settings(max_examples=200)
@given(
    order=st.integers(min_value=1, max_value=10),
    wn=st.floats(min_value=0.01, max_value=0.99)
)
def test_butter_filter_stability(order, wn):
    b, a = scipy.signal.butter(order, wn)
    z, p, k = scipy.signal.tf2zpk(b, a)

    pole_magnitudes = np.abs(p)
    assert np.all(pole_magnitudes < 1.0), \
        f"Butterworth filter should be stable (all poles inside unit circle), but max pole magnitude = {np.max(pole_magnitudes)}"
```

**Failing input**: `order=10, wn=0.015625`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

order = 10
wn = 0.015625

b, a = scipy.signal.butter(order, wn)
z, p, k = scipy.signal.tf2zpk(b, a)

pole_magnitudes = np.abs(p)
max_pole_mag = np.max(pole_magnitudes)

print(f"Butterworth filter (order={order}, wn={wn})")
print(f"Max pole magnitude: {max_pole_mag}")
print(f"Poles outside unit circle: {np.sum(pole_magnitudes >= 1.0)}")
print(f"Filter is {'UNSTABLE' if max_pole_mag >= 1.0 else 'stable'}")
```

Output:
```
Butterworth filter (order=10, wn=0.015625)
Max pole magnitude: 1.0064305587673334
Poles outside unit circle: 3
Filter is UNSTABLE
```

The same issue occurs with `scipy.signal.cheby1`:
```python
import numpy as np
import scipy.signal

order = 9
rp = 1.0
wn = 0.015625

b, a = scipy.signal.cheby1(order, rp, wn)
z, p, k = scipy.signal.tf2zpk(b, a)

pole_magnitudes = np.abs(p)
print(f"Chebyshev Type I filter (order={order}, rp={rp}, wn={wn})")
print(f"Max pole magnitude: {np.max(pole_magnitudes)}")
print(f"Poles outside unit circle: {np.sum(pole_magnitudes >= 1.0)}")
```

Output:
```
Chebyshev Type I filter (order=9, rp=1.0, wn=0.015625)
Max pole magnitude: 1.0116928435934867
Poles outside unit circle: 4
```

## Why This Is A Bug

Digital filters must have all poles strictly inside the unit circle (magnitude < 1.0) to be stable. A stable filter guarantees that bounded inputs produce bounded outputs. Poles at or outside the unit circle lead to unstable behavior where the filter output can grow unbounded or oscillate indefinitely.

The bug occurs when designing high-order filters with very low cutoff frequencies. The filter design algorithm produces badly conditioned coefficients (as indicated by the `BadCoefficients` warning), leading to numerical errors that push some poles outside the unit circle.

This violates the fundamental contract of filter design functions - they should always produce stable filters.

## Fix

The issue stems from numerical conditioning problems when the bilinear transform is applied with extreme frequency warping. Potential fixes:

1. **Use second-order sections (SOS) format**: The `butter` and `cheby1` functions support `output='sos'` which is more numerically stable:

```python
sos = scipy.signal.butter(order, wn, output='sos')
```

2. **Add input validation**: Warn or error when filter parameters are likely to cause numerical issues.

3. **Improve numerical stability**: Use higher-precision arithmetic or more stable algorithms for extreme parameter values.

The recommended workaround for users is to always use SOS format for higher-order filters:
```python
sos = scipy.signal.butter(N, Wn, output='sos')
filtered = scipy.signal.sosfilt(sos, data)
```