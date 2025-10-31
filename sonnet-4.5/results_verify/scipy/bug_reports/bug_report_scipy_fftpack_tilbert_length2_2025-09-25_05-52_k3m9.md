# Bug Report: scipy.fftpack.tilbert/itilbert Round-Trip Failure for Length-2 Arrays

**Target**: `scipy.fftpack.tilbert` and `scipy.fftpack.itilbert`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Tilbert transform and its inverse fail to round-trip correctly for length-2 arrays with zero mean, similar to the known issue with `hilbert`/`ihilbert`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import fftpack


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=2, max_size=100))
def test_tilbert_itilbert_roundtrip(x_list):
    x = np.array(x_list)
    x = x - x.mean()
    assume(np.sum(np.abs(x)) > 1e-6)

    h_param = 0.5

    t = fftpack.tilbert(x, h=h_param)
    it = fftpack.itilbert(t, h=h_param)

    assert np.allclose(it, x, rtol=1e-3, atol=1e-5)
```

**Failing input**: `x_list=[0.0, 1.0]` (after mean subtraction: `[-0.5, 0.5]`)

## Reproducing the Bug

```python
import numpy as np
from scipy import fftpack

x = np.array([0.0, 1.0])
x = x - x.mean()

print(f"Input (zero-mean): {x}")
print(f"sum(x) = {np.sum(x)}")

h = 0.5
print(f"h parameter: {h}")

t = fftpack.tilbert(x, h=h)
print(f"\ntilbert(x, h={h}) = {t}")

it = fftpack.itilbert(t, h=h)
print(f"itilbert(tilbert(x), h={h}) = {it}")

print(f"\nExpected: {x}")
print(f"Actual: {it}")
print(f"Match: {np.allclose(it, x, rtol=1e-3, atol=1e-5)}")

print("\n" + "="*60)
print("For comparison, length-3 works correctly:")
x3 = np.array([0.0, 1.0, 2.0])
x3 = x3 - x3.mean()
print(f"x3 = {x3}, sum = {np.sum(x3)}")
t3 = fftpack.tilbert(x3, h=h)
it3 = fftpack.itilbert(t3, h=h)
print(f"itilbert(tilbert(x3), h={h}) = {it3}")
print(f"Match: {np.allclose(it3, x3, rtol=1e-3, atol=1e-5)}")
```

**Output**:
```
Input (zero-mean): [-0.5  0.5]
sum(x) = 0.0
h parameter: 0.5

tilbert(x, h=0.5) = [0. 0.]
itilbert(tilbert(x), h=0.5) = [0. 0.]

Expected: [-0.5  0.5]
Actual: [0. 0.]
Match: False

============================================================
For comparison, length-3 works correctly:
x3 = [-1.  0.  1.], sum = 0.0
itilbert(tilbert(x3), h=0.5) = [-1.  0.  1.]
Match: True
```

## Why This Is A Bug

Like `hilbert`/`ihilbert`, the `tilbert`/`itilbert` pair should preserve the input signal when composed. For length-2 zero-mean arrays, the transformation zeros out the signal completely, causing information loss.

This is the same underlying issue as the `hilbert` bug - special handling of Nyquist modes for even-length arrays (particularly length-2) causes information loss for these pseudo-differential operators.

## Fix

This bug shares the same root cause as the `hilbert`/`ihilbert` bug. The fix would likely be in `/scipy/fftpack/_pseudo_diffs.py` where `tilbert` and `itilbert` are implemented. The Nyquist mode handling for length-2 (and potentially other even-length) arrays needs to be adjusted to preserve invertibility for zero-mean inputs.