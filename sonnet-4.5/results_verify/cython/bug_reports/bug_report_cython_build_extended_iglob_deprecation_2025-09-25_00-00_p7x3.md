# Bug Report: Cython.Build.Dependencies.extended_iglob Deprecation Warning

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `extended_iglob` function uses deprecated positional argument syntax for `re.split()` that generates `DeprecationWarning` in Python 3.13+.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import extended_iglob
import warnings


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/*', min_size=5, max_size=50))
@settings(max_examples=500)
def test_extended_iglob_no_warnings(pattern):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(extended_iglob(pattern))
        assert len(w) == 0, f"extended_iglob should not generate warnings: {[str(x.message) for x in w]}"
```

**Failing input**: Any pattern containing `**/`, such as `'**/*.py'`

## Reproducing the Bug

```python
import warnings
from Cython.Build.Dependencies import extended_iglob

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    list(extended_iglob('**/*.py'))
    for warning in w:
        print(f'{warning.category.__name__}: {warning.message}')
```

**Output**:
```
DeprecationWarning: 'maxsplit' is passed as positional argument
```

## Why This Is A Bug

Python 3.13 deprecates passing `maxsplit` as a positional argument to `re.split()`. While the code still works, it generates deprecation warnings and will break in future Python versions.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -52,7 +52,7 @@ def extended_iglob(pattern):
     # because '/' is generally common for relative paths.
     if '**/' in pattern or os.sep == '\\' and '**\\' in pattern:
         seen = set()
-        first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, 1)
+        first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, maxsplit=1)
         if first:
             first = iglob(first + os.sep)
         else:
```