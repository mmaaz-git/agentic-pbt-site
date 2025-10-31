# Bug Report: pandas.util.version._parse_letter_version Integer Zero Bug

**Target**: `pandas.util.version._parse_letter_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_parse_letter_version` function incorrectly treats integer `0` as falsy, returning `None` instead of `('post', 0)` when called with `_parse_letter_version(None, 0)`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas.util.version as version_module

@given(st.integers(min_value=0, max_value=1))
@settings(max_examples=1000)
def test_parse_letter_version_integer_zero_bug(number):
    result = version_module._parse_letter_version(None, number)

    if number == 0:
        assert result == ("post", 0), f"Bug: integer 0 treated as falsy, returns {result}"
```

**Failing input**: `number=0`

## Reproducing the Bug

```python
import pandas.util.version as version_module

result = version_module._parse_letter_version(None, 0)
print(f"_parse_letter_version(None, 0) = {result}")
print(f"Expected: ('post', 0)")
print(f"Actual: {result}")

result_string = version_module._parse_letter_version(None, "0")
print(f"\n_parse_letter_version(None, '0') = {result_string}")
print("Inconsistent behavior: string '0' works but integer 0 doesn't")
```

## Why This Is A Bug

The function signature accepts `number: str | bytes | SupportsInt`, indicating that integers are valid inputs. However, the condition `if not letter and number:` treats integer `0` as falsy, causing it to return `None` instead of the expected `('post', 0)`. This violates:

1. Type contract (function accepts integers but handles them incorrectly)
2. Consistency (integer `0` behaves differently from integer `1` and string `"0"`)
3. PEP 440 semantics (version "1.0-0" should be valid, meaning post-release 0)

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -123,7 +123,7 @@ def _parse_letter_version(
         return letter, int(number)
-    if not letter and number:
+    if not letter and number is not None:
         # We assume if we are given a number, but we are not given a letter
         # then this is using the implicit post release syntax (e.g. 1.0-1)
         letter = "post"
```