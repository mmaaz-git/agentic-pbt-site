# Bug Report: llm.default_plugins.openai_models.redact_data Mutates Input Dictionary

**Target**: `llm.default_plugins.openai_models.redact_data`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `redact_data()` function mutates its input dictionary in-place instead of creating a copy, violating Python conventions for data transformation functions and potentially causing unexpected side effects when input data is reused elsewhere in the program.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis property-based test demonstrating the input mutation bug in redact_data()
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import redact_data
import copy


@given(st.text(min_size=1))
def test_redact_data_should_not_mutate_input_image_url(data_content):
    """Test that redact_data does not mutate its input dictionary."""
    original = {"image_url": {"url": f"data:image/png;base64,{data_content}"}}
    original_copy = copy.deepcopy(original)

    result = redact_data(original)

    assert original == original_copy, (
        f"redact_data mutated its input! "
        f"Before: {original_copy}, After: {original}"
    )


if __name__ == "__main__":
    # Run the test
    test_redact_data_should_not_mutate_input_image_url()
```

<details>

<summary>
**Failing input**: `data_content='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 29, in <module>
    test_redact_data_should_not_mutate_input_image_url()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 14, in test_redact_data_should_not_mutate_input_image_url
    def test_redact_data_should_not_mutate_input_image_url(data_content):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 21, in test_redact_data_should_not_mutate_input_image_url
    assert original == original_copy, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: redact_data mutated its input! Before: {'image_url': {'url': 'data:image/png;base64,0'}}, After: {'image_url': {'url': 'data:...'}}
Falsifying example: test_redact_data_should_not_mutate_input_image_url(
    data_content='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case demonstrating the input mutation bug in redact_data()
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import redact_data
import copy

# Create a test dictionary with image data
original = {
    "image_url": {"url": "data:image/png;base64,abc123"}
}

# Create a deep copy to preserve the original state
original_copy = copy.deepcopy(original)

# Call redact_data - this should not mutate the input
result = redact_data(original)

# Test 1: Check if the original was mutated (it shouldn't be)
print("Test 1: Check if original was mutated")
print(f"Original before: {original_copy}")
print(f"Original after:  {original}")
print(f"Are they equal? {original == original_copy}")
print()

# Test 2: Check what the original looks like now
print("Test 2: Verify the mutation")
print(f"Original is now: {original}")
print(f"Expected mutation: {{'image_url': {{'url': 'data:...'}}}}")
print(f"Matches expected mutation? {original == {'image_url': {'url': 'data:...'}}}")
print()

# Test 3: Check if the result is the same object as the input
print("Test 3: Check if result is same object as input")
print(f"Result is original? {result is original}")
print(f"Result id: {id(result)}, Original id: {id(original)}")
```

<details>

<summary>
Output demonstrates input mutation and same object reference
</summary>
```
Test 1: Check if original was mutated
Original before: {'image_url': {'url': 'data:image/png;base64,abc123'}}
Original after:  {'image_url': {'url': 'data:...'}}
Are they equal? False

Test 2: Verify the mutation
Original is now: {'image_url': {'url': 'data:...'}}
Expected mutation: {'image_url': {'url': 'data:...'}}
Matches expected mutation? True

Test 3: Check if result is same object as input
Result is original? True
Result id: 136763634532288, Original id: 136763634532288
```
</details>

## Why This Is A Bug

This violates expected behavior for data transformation functions in Python. The function directly mutates nested dictionary values at lines 982 and 984 in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`:

```python
value["url"] = "data:..."  # Line 982 - mutates input!
value["data"] = "..."      # Line 984 - mutates input!
```

Key issues:
1. **Violates Python conventions**: Data transformation functions should return new objects unless explicitly documented as mutating in-place
2. **Function name is misleading**: "redact_data" implies transformation that returns redacted data, not in-place mutation
3. **Causes unexpected side effects**: When the same dictionary is used elsewhere, it gets unexpectedly modified
4. **Returns the same object**: The function returns `input_dict` making it appear like it might be creating a new copy when it's actually returning the mutated original
5. **Documentation is ambiguous**: The docstring says "modify" but doesn't explicitly state whether this means in-place mutation or creating a modified copy

## Relevant Context

The function is located in the `default_plugins` module, suggesting it's part of the public API. While current usage in the codebase always passes fresh dictionaries (e.g., `redact_data({"messages": messages})`), this doesn't excuse the underlying design flaw. The bug can manifest when:

- Users call the function with reusable data structures
- The same dictionary needs to be used for logging before and after redaction
- Multiple redaction passes are needed with different configurations
- Testing scenarios where the original data needs to be preserved

The function is used for redacting sensitive data (base64-encoded images and audio) from API request/response logging, making correctness particularly important.

## Proposed Fix

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