# Bug Report: pandas.core.dtypes.inference.is_re_compilable crashes on invalid regex patterns

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function crashes with `re.PatternError` when given invalid regex patterns like `"("`, `")"`, `"?"`, or `"*"`, instead of returning `False` as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import re
from pandas.core.dtypes.inference import is_re_compilable


@given(
    pattern=st.one_of(
        st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=10),
        st.integers(),
        st.floats(),
        st.none(),
    )
)
def test_is_re_compilable_consistent_with_re_compile(pattern):
    result = is_re_compilable(pattern)

    try:
        re.compile(pattern)
        can_compile = True
    except (TypeError, re.error):
        can_compile = False

    if can_compile:
        assert result, f"is_re_compilable({pattern!r}) returned False but re.compile succeeded"
```

**Failing input**: `'('`, `')'`, `'?'`, `'*'`, `'['`

## Reproducing the Bug

```python
from pandas.core.dtypes.inference import is_re_compilable

is_re_compilable("(")
```

This raises:
```
re.PatternError: missing ), unterminated subpattern at position 0
```

## Why This Is A Bug

According to the docstring, `is_re_compilable` should:
> Check if the object can be compiled into a regex pattern instance.
> Returns: bool - Whether `obj` can be compiled as a regex pattern.

The function should return `False` for invalid regex patterns, not crash. The current implementation only catches `TypeError` (for non-string inputs) but not `re.error`/`re.PatternError` (for invalid regex syntax).

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -186,7 +186,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```