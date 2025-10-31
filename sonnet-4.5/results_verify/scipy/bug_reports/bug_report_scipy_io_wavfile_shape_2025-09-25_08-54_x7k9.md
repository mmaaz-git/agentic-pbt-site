# Bug Report: scipy.io.wavfile Round-Trip Shape Violation

**Target**: `scipy.io.wavfile.write` and `scipy.io.wavfile.read`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing mono audio data with shape `(N, 1)` and reading it back, the shape changes to `(N,)`, violating the round-trip property that holds for all other channel counts.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import tempfile
import scipy.io.wavfile

@given(
    sample_rate=st.integers(min_value=1000, max_value=48000),
    n_samples=st.integers(min_value=1, max_value=5000),
    n_channels=st.integers(min_value=1, max_value=8),
    dtype=st.sampled_from([np.int16, np.int32, np.float32])
)
@settings(max_examples=200)
def test_wavfile_roundtrip_multichannel(sample_rate, n_samples, n_channels, dtype):
    if dtype in [np.int16, np.int32]:
        iinfo = np.iinfo(dtype)
        data = np.random.randint(iinfo.min, iinfo.max + 1, (n_samples, n_channels), dtype=dtype)
    else:
        data = (np.random.rand(n_samples, n_channels) * 2 - 1).astype(dtype)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name

    try:
        scipy.io.wavfile.write(temp_file, sample_rate, data)
        loaded_rate, loaded_data = scipy.io.wavfile.read(temp_file)

        assert loaded_data.shape == data.shape, f"Shape mismatch: {loaded_data.shape} != {data.shape}"
    finally:
        os.unlink(temp_file)
```

**Failing input**: `n_channels=1` with any sample_rate, n_samples, and dtype

## Reproducing the Bug

```python
import numpy as np
import tempfile
from scipy.io import wavfile

data_2d = np.array([[100]], dtype=np.int16)

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    wavfile.write(f.name, 1000, data_2d)
    rate, loaded = wavfile.read(f.name)

print(f"Written: {data_2d.shape}")
print(f"Read:    {loaded.shape}")

assert data_2d.shape == loaded.shape, f"Shape mismatch: wrote {data_2d.shape}, read {loaded.shape}"
```

Output:
```
Written: (1, 1)
Read:    (1,)
AssertionError: Shape mismatch: wrote (1, 1), read (1,)
```

## Why This Is A Bug

1. **Violates round-trip property**: `read(write(data))` should return data with the same shape
2. **Inconsistent behavior**:
   - `(N, 1)` → reads as `(N,)` ✗
   - `(N, 2)` → reads as `(N, 2)` ✓
   - `(N, 3)` → reads as `(N, 3)` ✓
3. **Documentation conflict**: The docs state "To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels)", which suggests `(N, 1)` is valid input
4. **User confusion**: Users cannot reliably preserve array shapes through WAV file I/O

## Fix

The issue is likely in `wavfile.read()` where mono audio is being squeezed from `(N, 1)` to `(N,)`. The fix should either:

**Option 1 (Recommended)**: Preserve the exact shape written - if `(N, 1)` was written, return `(N, 1)`

**Option 2**: Document that mono audio is always returned as `(N,)` regardless of input shape, and adjust `write()` to handle this consistently

The recommended fix would be in `wavfile.read()` to check if the original data was written with `nchannels=1` and preserve the 2D shape if that's what was written. However, this may require format changes or metadata to distinguish between intentional `(N,)` and squeezed `(N, 1)` arrays.