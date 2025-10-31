# Bug Report: django.urls.IntConverter Accepts Invalid Input

**Target**: `django.urls.converters.IntConverter`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

IntConverter.to_python() accepts negative integers despite its regex pattern `[0-9]+` which only matches non-negative integers, violating the converter's contract.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st
from django.urls.converters import IntConverter
import pytest


@given(st.text(min_size=1, max_size=20))
def test_int_converter_contract_to_python_validates_regex(s):
    converter = IntConverter()
    regex = re.compile(f'^{converter.regex}$')

    if regex.match(s):
        result = converter.to_python(s)
        assert isinstance(result, int)
    else:
        with pytest.raises((ValueError, TypeError)):
            converter.to_python(s)
```

**Failing input**: `'-5'` and any negative number string

## Reproducing the Bug

```python
import re
from django.urls.converters import IntConverter

converter = IntConverter()
regex = re.compile(f'^{converter.regex}$')

negative_input = '-5'
assert not regex.match(negative_input)

result = converter.to_python(negative_input)
assert result == -5
```

## Why This Is A Bug

The converter's regex pattern defines its contract: what inputs are considered valid. The regex `[0-9]+` explicitly excludes negative numbers (no `-` sign allowed). However, `to_python()` uses Python's `int()` which accepts negative numbers.

This violates the API contract in two ways:

1. **Inconsistent validation**: The regex says "only non-negative integers" but `to_python()` accepts negatives
2. **Expected error handling**: In `resolvers.py:330-334`, ValueError from `to_python()` is expected to cause match failure, indicating that `to_python()` should validate its input

While this doesn't affect normal Django URL routing (negative numbers never reach `to_python()` because they don't match the regex), it creates confusion for anyone using converters programmatically or extending them.

## Fix

Add validation to `IntConverter.to_python()` to match the regex constraint:

```diff
--- a/django/urls/converters.py
+++ b/django/urls/converters.py
@@ -8,7 +8,11 @@ class IntConverter:
     regex = "[0-9]+"

     def to_python(self, value):
-        return int(value)
+        result = int(value)
+        if result < 0:
+            raise ValueError(
+                f"IntConverter does not accept negative values: {value}"
+            )
+        return result

     def to_url(self, value):
         return str(value)
```