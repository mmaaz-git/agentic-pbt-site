# Bug Report: scipy.io.wavfile Round-Trip Shape Inconsistency

**Target**: `scipy.io.wavfile.write` and `scipy.io.wavfile.read`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `wavfile.write()` and `wavfile.read()` functions do not preserve array shape for 2-D single-channel audio data in round-trip operations. When writing a 2-D array with shape `(n, 1)`, reading it back returns a 1-D array with shape `(n,)`, violating the round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io import wavfile
import tempfile
import numpy as np

@given(
    st.integers(min_value=8000, max_value=48000),
    st.integers(min_value=10, max_value=500),
    st.integers(min_value=1, max_value=8)
)
@settings(max_examples=20)
def test_wavfile_multichannel(rate, n_samples, n_channels):
    data = np.random.randint(-10000, 10000, size=(n_samples, n_channels), dtype=np.int16)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filename = f.name

    try:
        wavfile.write(filename, rate, data)
        result_rate, result_data = wavfile.read(filename)

        assert result_rate == rate
        assert result_data.shape == data.shape  # FAILS when n_channels == 1
        assert np.array_equal(result_data, data)
    finally:
        import os
        if os.path.exists(filename):
            os.unlink(filename)
```

**Failing input**: `rate=8000, n_samples=10, n_channels=1`

## Reproducing the Bug

```python
import tempfile
import numpy as np
from scipy.io import wavfile

rate = 44100
data_2d_single_channel = np.array([[100], [200], [300]], dtype=np.int16)

print(f"Original shape: {data_2d_single_channel.shape}")

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    filename = f.name

wavfile.write(filename, rate, data_2d_single_channel)
result_rate, result_data = wavfile.read(filename)

print(f"Result shape: {result_data.shape}")
print(f"Expected: {data_2d_single_channel.shape}, Got: {result_data.shape}")

import os
os.unlink(filename)
```

Output:
```
Original shape: (3, 1)
Result shape: (3,)
Expected: (3, 1), Got: (3,)
```

## Why This Is A Bug

This violates the expected round-trip property: `wavfile.read(wavfile.write(rate, data))` should return data with the same shape as the input. While the documentation states "Data is 1-D for 1-channel WAV", this creates an asymmetry:

- `wavfile.write()` accepts both 1-D arrays and 2-D arrays with shape `(n, 1)` for single-channel audio
- `wavfile.read()` always returns 1-D arrays for single-channel audio

This inconsistency can cause unexpected behavior in code that relies on preserving array shapes, particularly in pipelines that process both mono and stereo audio with consistent 2-D array representations.

## Fix

```diff
--- a/scipy/io/wavfile.py
+++ b/scipy/io/wavfile.py
@@ -534,8 +534,8 @@ def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, is_r

     _handle_pad_byte(fid, size)

-    if channels > 1:
-        data = data.reshape(-1, channels)
+    # Always reshape to (n_samples, n_channels) for consistency
+    data = data.reshape(-1, channels)
     return data
```

This change ensures that `wavfile.read()` always returns a 2-D array with shape `(n_samples, n_channels)`, even when `n_channels == 1`, maintaining consistency with the input format accepted by `wavfile.write()` and preserving the round-trip property.

Note: This is a breaking change that would affect existing code expecting 1-D arrays for mono audio. An alternative fix would be to update the documentation to explicitly warn users about this shape inconsistency.