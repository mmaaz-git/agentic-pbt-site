# Bug Report: llm.default_plugins.openai_models.redact_data Input Mutation

**Target**: `llm.default_plugins.openai_models.redact_data`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `redact_data()` function mutates its input dictionary instead of creating a copy, violating the principle of immutability and causing unexpected side effects when the input data is reused.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import redact_data
import copy


@given(st.text(min_size=1))
def test_redact_data_should_not_mutate_input_image_url(data_content):
    original = {"image_url": {"url": f"data:image/png;base64,{data_content}"}}
    original_copy = copy.deepcopy(original)

    result = redact_data(original)

    assert original == original_copy, (
        f"redact_data mutated its input! "
        f"Before: {original_copy}, After: {original}"
    )
```

**Failing input**: Any string, e.g., `data_content = "abc123"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import redact_data
import copy

original = {
    "image_url": {"url": "data:image/png;base64,abc123"}
}

original_copy = copy.deepcopy(original)
result = redact_data(original)

assert original != original_copy
assert original == {"image_url": {"url": "data:..."}}
assert result is original
```

## Why This Is A Bug

The `redact_data()` function modifies the input dictionary in place (lines 982 and 984 in openai_models.py):

```python
value["url"] = "data:..."  # Mutates the input!
value["data"] = "..."      # Mutates the input!
```

This violates the expected behavior of a data processing function. When `redact_data()` is called with data structures that may be reused elsewhere in the program, those structures are unexpectedly modified. This can lead to subtle bugs where data that should remain unchanged is accidentally redacted.

The function returns `input_dict` at the end, and callers may assume this is a new copy when it's actually the same mutated object.

## Fix

Create a deep copy of the input before processing, or build a new structure while recursing:

```diff
 def redact_data(input_dict):
     """
     Recursively search through the input dictionary for any 'image_url' keys
     and modify the 'url' value to be just 'data:...'.

     Also redact input_audio.data keys
     """
     if isinstance(input_dict, dict):
+        result = {}
         for key, value in input_dict.items():
             if (
                 key == "image_url"
                 and isinstance(value, dict)
                 and "url" in value
                 and value["url"].startswith("data:")
             ):
-                value["url"] = "data:..."
+                result[key] = {**value, "url": "data:..."}
             elif key == "input_audio" and isinstance(value, dict) and "data" in value:
-                value["data"] = "..."
+                result[key] = {**value, "data": "..."}
             else:
-                redact_data(value)
+                result[key] = redact_data(value)
+        return result
     elif isinstance(input_dict, list):
-        for item in input_dict:
-            redact_data(item)
+        return [redact_data(item) for item in input_dict]
-    return input_dict
+    else:
+        return input_dict
```