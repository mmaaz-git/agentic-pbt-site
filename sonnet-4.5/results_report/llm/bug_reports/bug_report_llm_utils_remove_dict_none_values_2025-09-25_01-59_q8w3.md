# Bug Report: llm.utils.remove_dict_none_values Inconsistent List Handling

**Target**: `llm.utils.remove_dict_none_values`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_dict_none_values` function inconsistently handles None values and empty dicts when they appear in lists versus when they appear as direct dict values. Empty dicts that result from removing None values are filtered out for direct dict values but preserved inside lists.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from llm.utils import remove_dict_none_values

@st.composite
def dict_with_nested_nones(draw):
    return {
        "direct_none": None,
        "direct_empty_dict": {"nested_none": None},
        "list_with_none": [None, 1, 2],
        "list_with_empty_dict": [{"nested_none": None}],
    }

@settings(max_examples=100)
@given(dict_with_nested_nones())
def test_remove_dict_none_values_consistency(d):
    result = remove_dict_none_values(d)

    def has_empty_dict(obj, path=""):
        if isinstance(obj, dict):
            if not obj:
                return True, path
            for k, v in obj.items():
                found, p = has_empty_dict(v, f"{path}.{k}")
                if found:
                    return True, p
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                found, p = has_empty_dict(item, f"{path}[{i}]")
                if found:
                    return True, p
        return False, ""

    found, path = has_empty_dict(result)
    assert not found, f"Empty dict found at {path} after remove_dict_none_values"
```

**Failing input**: `{"a": [{"b": None}]}`

## Reproducing the Bug

```python
from llm.utils import remove_dict_none_values

d = {"direct": {"x": None}, "in_list": [{"y": None}]}
result = remove_dict_none_values(d)

print(f"Input:  {d}")
print(f"Output: {result}")
```

**Output:**
```
Input:  {'direct': {'x': None}, 'in_list': [{'y': None}]}
Output: {'in_list': [{}]}
```

The function removes the empty dict `{"x": None}` when it's a direct value, but preserves the empty dict `{}` (which came from `{"y": None}`) when it's inside a list.

Additional issues:
- `remove_dict_none_values({"a": [1, None, 2]})` returns `{"a": [1, None, 2]}` - None values in lists are not removed
- `remove_dict_none_values({"a": [{"b": None}]})` returns `{"a": [{}]}` - empty dicts in lists are not removed

## Why This Is A Bug

The function's docstring states it should "Recursively remove keys with value of None or value of a dict that is all values of None". The implementation handles this correctly for direct dict values (lines 87-90) by filtering out empty nested dicts:

```python
if isinstance(value, dict):
    nested = remove_dict_none_values(value)
    if nested:  # Empty dicts are filtered out
        new_dict[key] = nested
```

However, for lists (line 92), the function simply maps `remove_dict_none_values` over list elements without filtering:

```python
elif isinstance(value, list):
    new_dict[key] = [remove_dict_none_values(v) for v in value]
```

This creates inconsistent behavior where:
1. Empty dicts resulting from None-removal are kept in lists but removed as direct values
2. None values in lists are preserved rather than removed

This inconsistency can lead to unexpected data structures in the output, potentially causing issues for downstream code that expects consistent None-filtering behavior.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -89,7 +89,15 @@ def remove_dict_none_values(d):
                 if nested:
                     new_dict[key] = nested
             elif isinstance(value, list):
-                new_dict[key] = [remove_dict_none_values(v) for v in value]
+                # Filter out None values and empty dicts from lists
+                cleaned_list = []
+                for v in value:
+                    cleaned = remove_dict_none_values(v)
+                    # Skip None values and empty dicts/lists
+                    if cleaned is not None and cleaned != {} and cleaned != []:
+                        cleaned_list.append(cleaned)
+                if cleaned_list:
+                    new_dict[key] = cleaned_list
             else:
                 new_dict[key] = value
     return new_dict
```

This fix ensures consistent behavior by filtering out None values and empty containers from lists, matching the behavior for direct dict values.