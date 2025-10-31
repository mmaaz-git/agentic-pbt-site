# Bug Report: scipy.io.matlab.savemat Crashes on Non-ASCII Variable Names

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `savemat` function crashes with `UnicodeEncodeError` when given variable names containing characters outside the latin-1 encoding range (Unicode codepoints > 255), instead of providing a helpful error message about invalid MATLAB variable names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import savemat, loadmat
import numpy as np
import tempfile
import os

@st.composite
def valid_varnames(draw):
    first_char = draw(st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_'),
        min_size=0, max_size=10
    ))
    return first_char + rest

@given(
    varname=valid_varnames(),
    value=st.floats(min_value=-1e10, max_value=1e10,
                   allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_valid_varnames_roundtrip(varname, value):
    arr = np.array([[value]])
    mdict = {varname: arr}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        savemat(fname, mdict, format='5')
        loaded = loadmat(fname)
        assert varname in loaded
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

if __name__ == "__main__":
    test_valid_varnames_roundtrip()
```

<details>

<summary>
**Failing input**: `varname='aĀ'` (contains Unicode character U+0100)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 38, in <module>
    test_valid_varnames_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 17, in test_valid_varnames_roundtrip
    varname=valid_varnames(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 30, in test_valid_varnames_roundtrip
    savemat(fname, mdict, format='5')
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio.py", line 317, in savemat
    MW.put_variables(mdict)
    ~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py", line 901, in put_variables
    self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
                                       ~~~~~~~~~~~^^^^^^^^^^
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 1: ordinal not in range(256)
Falsifying example: test_valid_varnames_roundtrip(
    varname='aĀ',
    value=0.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import tempfile
from scipy.io.matlab import savemat

arr = np.array([[1.0]])
mdict = {'aĀ': arr}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

savemat(fname, mdict, format='5')
```

<details>

<summary>
UnicodeEncodeError when encoding variable name with non-latin1 character
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 11, in <module>
    savemat(fname, mdict, format='5')
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio.py", line 317, in savemat
    MW.put_variables(mdict)
    ~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py", line 901, in put_variables
    self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
                                       ~~~~~~~~~~~^^^^^^^^^^
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 1: ordinal not in range(256)
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Inconsistent validation**: The function already validates and warns about variable names starting with underscores (line 884-887 in _mio5.py), showing that input validation is an established pattern. However, it fails to validate the more fundamental requirement that variable names must be encodable.

2. **Poor error handling**: Instead of providing a user-friendly error message explaining that MATLAB variable names must be ASCII-only (as per MATLAB's documented requirements), the function crashes with a low-level `UnicodeEncodeError` that doesn't guide users to the solution.

3. **Documentation gap**: The scipy.io.savemat documentation doesn't mention that variable names must be ASCII or latin-1 compatible, yet the function attempts to encode them as latin-1 without validation.

4. **MATLAB compatibility mismatch**: MATLAB only supports ASCII variable names (letters A-Z, a-z, digits 0-9, and underscore), but scipy attempts to encode with latin-1 (which supports 256 characters). Even this broader encoding fails for Unicode characters beyond codepoint 255.

## Relevant Context

The crash occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py` at lines 893 and 901:

```python
# Line 893 (with compression):
self._matrix_writer.write_top(var, name.encode('latin1'), is_global)

# Line 901 (without compression):
self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
```

MATLAB's variable naming rules (from MathWorks documentation):
- Must begin with a letter (A-Z, a-z)
- Can contain letters, digits, and underscores
- Only US-ASCII characters are supported
- Maximum length is 63 characters

Related scipy code showing existing validation pattern (lines 884-887):
```python
if name[0] == '_':
    msg = (f"Starting field name with a "
           f"underscore ({name}) is ignored")
    warnings.warn(msg, MatWriteWarning, stacklevel=2)
    continue
```

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -882,6 +882,15 @@ class MatFile5Writer:
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
+            # Validate variable name can be encoded as ASCII (MATLAB requirement)
+            try:
+                name_encoded = name.encode('ascii')
+            except UnicodeEncodeError:
+                raise MatWriteError(
+                    f"MATLAB variable names must contain only ASCII characters. "
+                    f"Variable name '{name}' contains non-ASCII characters."
+                )
+
             if name[0] == '_':
                 msg = (f"Starting field name with a "
                        f"underscore ({name}) is ignored")
@@ -891,11 +900,11 @@ class MatFile5Writer:
             if self.do_compression:
                 stream = BytesIO()
                 self._matrix_writer.file_stream = stream
-                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
+                self._matrix_writer.write_top(var, name_encoded, is_global)
                 out_str = zlib.compress(stream.getvalue())
                 tag = np.empty((), NDT_TAG_FULL)
                 tag['mdtype'] = miCOMPRESSED
                 tag['byte_count'] = len(out_str)
                 self.file_stream.write(tag.tobytes())
                 self.file_stream.write(out_str)
             else:  # not compressing
-                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
+                self._matrix_writer.write_top(var, name_encoded, is_global)
```