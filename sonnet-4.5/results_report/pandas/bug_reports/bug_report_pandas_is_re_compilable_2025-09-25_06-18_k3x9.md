# Bug Report: pandas.core.dtypes.inference.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is documented to check if an object can be compiled into a regex pattern, returning a boolean. However, it crashes with `re.PatternError` when given invalid regex patterns (e.g., `'['`, `'('`, `'?'`) instead of returning `False`.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.inference import is_re_compilable

@given(text=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=20))
@settings(max_examples=500)
def test_is_re_compilable_should_not_crash(text):
    try:
        result = is_re_compilable(text)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    except re.PatternError as e:
        raise AssertionError(f"is_re_compilable should not raise PatternError for invalid patterns, but got: {e}")
```

**Failing inputs**: `'['`, `'('`, `')'`, `'?'`, `'*'`, `'+'`, `'\'`, and other invalid regex patterns

## Reproducing the Bug

```python
import re
from pandas.core.dtypes.inference import is_re_compilable

pattern = '['

result = is_re_compilable(pattern)
```

**Output**:
```
re.PatternError: unterminated character set at position 0
```

**Expected**: The function should return `False` instead of raising an exception.

## Why This Is A Bug

The function's docstring and name (`is_re_compilable`) imply it checks whether a pattern can be compiled, returning a boolean result. The current implementation only catches `TypeError`, but invalid regex patterns raise `re.PatternError`, which propagates to the caller. This violates the function's contract and causes crashes in dependent code like `pandas.core.array_algos.replace.should_use_regex`.

## Fix

```diff
diff --git a/pandas/core/dtypes/inference.py b/pandas/core/dtypes/inference.py
index abc123..def456 100644
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     False
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.PatternError):
         return False
     else:
         return True
```