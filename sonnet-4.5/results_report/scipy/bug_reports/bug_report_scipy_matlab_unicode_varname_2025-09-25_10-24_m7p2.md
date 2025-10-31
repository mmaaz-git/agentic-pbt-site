# Bug Report: scipy.io.matlab savemat Crashes on Non-Latin1 Variable Names

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `savemat` function crashes with `UnicodeEncodeError` when given variable names containing characters outside the latin-1 encoding range, instead of failing gracefully with a helpful error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import savemat
import numpy as np
import tempfile

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
```

**Failing input**: `varname='aĀ'` (contains character '\u0100' which is outside latin-1)

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

Output:
```
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 1: ordinal not in range(256)
```

## Why This Is A Bug

MATLAB variable names must be valid ASCII identifiers, but the current implementation crashes with an unhelpful encoding error instead of validating the variable name and providing a clear error message. The crash occurs in `_mio5.py:893` and `_mio5.py:901`:

```python
self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
```

This fails for any character outside the latin-1 range, which is a larger set than valid MATLAB variable names anyway (MATLAB only allows ASCII letters, digits, and underscores).

## Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,6 +881,15 @@ class MatFile5Writer:
             self.write_file_header()
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
+            # Validate variable name can be encoded
+            try:
+                name_encoded = name.encode('ascii')
+            except UnicodeEncodeError:
+                raise MatWriteError(
+                    f"MATLAB variable names must be ASCII. "
+                    f"Variable name '{name}' contains non-ASCII characters."
+                )
+
             if name[0] == '_':
                 msg = (f"Starting field name with a "
                        f"underscore ({name}) is ignored")
@@ -890,11 +899,11 @@ class MatFile5Writer:
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
             else:
-                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
+                self._matrix_writer.write_top(var, name_encoded, is_global)
```