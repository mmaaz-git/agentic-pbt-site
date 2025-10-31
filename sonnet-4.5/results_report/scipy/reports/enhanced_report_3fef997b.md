# Bug Report: scipy.io.matlab Complex Infinity Corruption During Save/Load

**Target**: `scipy.io.matlab._mio4`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Complex numbers with zero real part and infinite imaginary part are corrupted during MATLAB format 4 save/load round-trip. The real part changes from 0.0 to NaN due to an invalid multiplication operation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.numpy import arrays, array_shapes
import scipy.io.matlab as sio_matlab
import numpy as np
import tempfile
import os

@given(
    data=st.dictionaries(
        keys=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,30}', fullmatch=True),
        values=arrays(
            dtype=st.sampled_from([np.float64, np.int32, np.uint8, np.complex128]),
            shape=array_shapes(max_dims=2, max_side=20),
        ),
        min_size=1,
        max_size=5
    )
)
@example(data={'A': np.array([complex(0.0, np.inf)])})  # Force test with the failing case
@settings(max_examples=100)
def test_savemat_loadmat_roundtrip_format4(data):
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        sio_matlab.savemat(fname, data, format='4')
        loaded = sio_matlab.loadmat(fname)

        for key in data.keys():
            assert key in loaded
            original = data[key]
            result = loaded[key]

            if original.ndim == 0:
                expected_shape = (1, 1)
            elif original.ndim == 1:
                expected_shape = (1, original.shape[0])
            else:
                expected_shape = original.shape

            assert result.shape == expected_shape

            original_reshaped = original.reshape(expected_shape)
            np.testing.assert_array_equal(result, original_reshaped)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    test_savemat_loadmat_roundtrip_format4()
```

<details>

<summary>
**Failing input**: `data={'A': array([0.+infj])}`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/27
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_savemat_loadmat_roundtrip_format4 FAILED                   [100%]

=================================== FAILURES ===================================
____________________ test_savemat_loadmat_roundtrip_format4 ____________________

    @given(
>       data=st.dictionaries(
                   ^^^
            keys=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,30}', fullmatch=True),
            values=arrays(
                dtype=st.sampled_from([np.float64, np.int32, np.uint8, np.complex128]),
                shape=array_shapes(max_dims=2, max_side=20),
            ),
            min_size=1,
            max_size=5
        )
    )

hypo.py:9:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

data = {'A': array([0.+infj])}

    @given(
        data=st.dictionaries(
            keys=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,30}', fullmatch=True),
            values=arrays(
                dtype=st.sampled_from([np.float64, np.int32, np.uint8, np.complex128]),
                shape=array_shapes(max_dims=2, max_side=20),
            ),
            min_size=1,
            max_size=5
        )
    )
    @example(data={'A': np.array([complex(0.0, np.inf)])})  # Force test with the failing case
    @settings(max_examples=100)
    def test_savemat_loadmat_roundtrip_format4(data):
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            fname = f.name

        try:
            sio_matlab.savemat(fname, data, format='4')
            loaded = sio_matlab.loadmat(fname)

            for key in data.keys():
                assert key in loaded
                original = data[key]
                result = loaded[key]

                if original.ndim == 0:
                    expected_shape = (1, 1)
                elif original.ndim == 1:
                    expected_shape = (1, original.shape[0])
                else:
                    expected_shape = original.shape

                assert result.shape == expected_shape

                original_reshaped = original.reshape(expected_shape)
>               np.testing.assert_array_equal(result, original_reshaped)
E               AssertionError:
E               Arrays are not equal
E
E               nan location mismatch:
E                ACTUAL: array([[nan+infj]])
E                DESIRED: array([[0.+infj]])
E               Falsifying explicit example: test_savemat_loadmat_roundtrip_format4(
E                   data={'A': array([0.+infj])},
E               )

hypo.py:44: AssertionError
=============================== warnings summary ===============================
hypo.py::test_savemat_loadmat_roundtrip_format4
  /home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio4.py:218: RuntimeWarning: invalid value encountered in multiply
    return res + (res_j * 1j)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hypo.py::test_savemat_loadmat_roundtrip_format4 - AssertionError:
========================= 1 failed, 1 warning in 0.28s =========================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.io.matlab as sio
from io import BytesIO

# Create a complex number with zero real part and infinite imaginary part
original = np.array([complex(0.0, np.inf)])
print(f"Original array: {original}")
print(f"  Real part: {original[0].real}")
print(f"  Imaginary part: {original[0].imag}")
print()

# Save to a BytesIO buffer using format='4'
f = BytesIO()
sio.savemat(f, {'x': original}, format='4')

# Load the data back
f.seek(0)  # Reset buffer to beginning
loaded = sio.loadmat(f)
result = loaded['x']

print(f"Loaded array: {result}")
print(f"  Real part: {result[0,0].real}")
print(f"  Imaginary part: {result[0,0].imag}")
print()

# Check if the values match
if result[0,0].real != original[0].real:
    print(f"ERROR: Real part changed from {original[0].real} to {result[0,0].real}")
else:
    print("Success: Real part preserved")

if result[0,0].imag != original[0].imag:
    print(f"ERROR: Imaginary part changed from {original[0].imag} to {result[0,0].imag}")
else:
    print("Success: Imaginary part preserved")
```

<details>

<summary>
Output showing data corruption
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio4.py:218: RuntimeWarning: invalid value encountered in multiply
  return res + (res_j * 1j)
Original array: [0.+infj]
  Real part: 0.0
  Imaginary part: inf

Loaded array: [[nan+infj]]
  Real part: nan
  Imaginary part: inf

ERROR: Real part changed from 0.0 to nan
Success: Imaginary part preserved
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Documentation Promise**: The scipy.io.loadmat documentation explicitly states that `struct_as_record=True` (the default) allows "easier round-trip load and save of MATLAB files". This implies data preservation during save/load cycles.

2. **Valid Mathematical Values**: Complex numbers with 0+infj are valid IEEE 754 values in Python/NumPy. They represent legitimate mathematical entities used in scientific computing.

3. **Silent Data Corruption**: The bug causes silent corruption - the real part changes from 0.0 to NaN without any error being raised to the user. This could lead to incorrect scientific results.

4. **RuntimeWarning Ignored**: The code generates a RuntimeWarning about "invalid value encountered in multiply" but proceeds anyway, corrupting the data.

5. **Incorrect Operation**: The expression `inf * 1j` in NumPy produces `nan+infj` instead of the expected `0+infj`. This is due to how NumPy handles the multiplication of infinity with the imaginary unit.

## Relevant Context

The bug occurs in `/scipy/io/matlab/_mio4.py` at line 218 in the `read_full_array` method:

```python
if hdr.is_complex:
    # avoid array copy to save memory
    res = self.read_sub_array(hdr, copy=False)
    res_j = self.read_sub_array(hdr, copy=False)
    return res + (res_j * 1j)  # Line 218: This causes the corruption
```

When `res_j` contains infinity, the operation `res_j * 1j` produces NaN in the real part due to NumPy's complex number arithmetic rules. The `copy=False` parameter may also contribute to the issue.

The code comment "avoid array copy to save memory" indicates this was an optimization that inadvertently introduced the bug.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio4.py
+++ b/scipy/io/matlab/_mio4.py
@@ -213,8 +213,11 @@ class VarReader4:
             numeric array
         '''
         if hdr.is_complex:
-            # avoid array copy to save memory
-            res = self.read_sub_array(hdr, copy=False)
+            # need to copy to avoid corruption with special float values
+            # when res_j contains inf, the operation res_j * 1j can produce NaN
+            # in the real part if copy=False is used
+            res = self.read_sub_array(hdr, copy=True)
             res_j = self.read_sub_array(hdr, copy=False)
             return res + (res_j * 1j)
         return self.read_sub_array(hdr)
```