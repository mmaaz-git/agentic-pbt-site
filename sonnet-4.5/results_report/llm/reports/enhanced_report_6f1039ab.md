# Bug Report: llm.default_plugins.openai_models not_nulls Function Crashes With Dict Arguments

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls()` function crashes with a ValueError when given a dict argument, which is how it's called in production code. The function expects an iterable of (key, value) tuples but receives a dict, causing unpacking errors when iterating.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovers the not_nulls bug."""

from hypothesis import given, strategies as st, settings, example
from llm.default_plugins.openai_models import not_nulls

@settings(max_examples=100)
@given(st.dictionaries(
    st.text(min_size=3, max_size=20),  # Use min_size=3 to avoid 2-char edge case
    st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.text()),
    min_size=1
))
@example(options_dict={"temperature": 0.7, "max_tokens": None})  # A specific failing example
def test_not_nulls_crashes_with_dict_argument(options_dict):
    """Test that demonstrates not_nulls crashes when given a dict instead of dict.items()

    The not_nulls function has a bug: it expects an iterable of (key, value) tuples
    but is called with dicts. When iterating over a dict, Python yields keys (strings),
    not (key, value) pairs. The function tries to unpack each key string into
    two variables, causing a ValueError.
    """
    # This should work correctly but doesn't due to the bug
    expected = {k: v for k, v in options_dict.items() if v is not None}

    # Try calling the buggy function
    try:
        result = not_nulls(options_dict)
        # If we get here with a non-2-char key dict, the bug is fixed
        assert all(len(k) == 2 for k in options_dict.keys()), \
            f"Bug seems fixed! Got {result}, expected crash for keys: {list(options_dict.keys())}"
    except ValueError as e:
        # This is the bug - it crashes instead of working
        assert "unpack" in str(e), f"Unexpected error: {e}"
        print(f"Bug confirmed! Function crashed with: {e}")
        print(f"  Input dict: {options_dict}")
        print(f"  Expected output: {expected}")
        raise  # Re-raise to show Hypothesis found the bug

if __name__ == "__main__":
    # Run the test
    test_not_nulls_crashes_with_dict_argument()
```

<details>

<summary>
**Failing input**: `{"temperature": 0.7, "max_tokens": None}`
</summary>
```
Bug confirmed! Function crashed with: too many values to unpack (expected 2)
  Input dict: {'temperature': 0.7, 'max_tokens': None}
  Expected output: {'temperature': 0.7}
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 41, in <module>
    test_not_nulls_crashes_with_dict_argument()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 8, in test_not_nulls_crashes_with_dict_argument
    @given(st.dictionaries(
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 27, in test_not_nulls_crashes_with_dict_argument
    result = not_nulls(options_dict)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/default_plugins/openai_models.py", line 916, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: too many values to unpack (expected 2)
Falsifying explicit example: test_not_nulls_crashes_with_dict_argument(
    options_dict={'temperature': 0.7, 'max_tokens': None},
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the not_nulls bug with dict argument."""

import sys
import traceback

# Import the function that has the bug
from llm.default_plugins.openai_models import not_nulls

# Test Case 1: Non-empty dict (this should crash)
print("Test 1: Non-empty dict with mixed values")
print("=" * 50)
options_dict = {"temperature": 0.7, "max_tokens": None, "top_p": 0.9}
print(f"Input: {options_dict}")
print(f"Type: {type(options_dict)}")

try:
    result = not_nulls(options_dict)
    print(f"Result: {result}")
except Exception as e:
    print(f"\nException raised: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc(file=sys.stdout)

print("\n" + "=" * 50)
print("\nTest 2: Empty dict (this should work)")
print("=" * 50)
empty_dict = {}
print(f"Input: {empty_dict}")
print(f"Type: {type(empty_dict)}")

try:
    result = not_nulls(empty_dict)
    print(f"Result: {result}")
except Exception as e:
    print(f"\nException raised: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc(file=sys.stdout)

print("\n" + "=" * 50)
print("\nTest 3: Dict.items() (this should work correctly)")
print("=" * 50)
options_dict = {"temperature": 0.7, "max_tokens": None, "top_p": 0.9}
print(f"Input: dict.items() from {options_dict}")
print(f"Type: {type(options_dict.items())}")

try:
    result = not_nulls(options_dict.items())
    print(f"Result: {result}")
except Exception as e:
    print(f"\nException raised: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc(file=sys.stdout)
```

<details>

<summary>
ValueError: too many values to unpack (expected 2)
</summary>
```
Test 1: Non-empty dict with mixed values
==================================================
Input: {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
Type: <class 'dict'>

Exception raised: ValueError
Error message: too many values to unpack (expected 2)

Full traceback:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/repo.py", line 18, in <module>
    result = not_nulls(options_dict)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/default_plugins/openai_models.py", line 916, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

==================================================

Test 2: Empty dict (this should work)
==================================================
Input: {}
Type: <class 'dict'>
Result: {}

==================================================

Test 3: Dict.items() (this should work correctly)
==================================================
Input: dict.items() from {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
Type: <class 'dict_items'>
Result: {'temperature': 0.7, 'top_p': 0.9}
```
</details>

## Why This Is A Bug

The `not_nulls()` function at line 915-916 in `/home/npc/miniconda/lib/python3.13/site-packages/llm/default_plugins/openai_models.py` has a critical implementation error:

```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}
```

This function is called at line 658 with `prompt.options`:
```python
kwargs = dict(not_nulls(prompt.options))
```

The bug occurs because:

1. **`prompt.options` can be a dict**: According to the Prompt class in models.py line 365, `self.options = options or {}`, meaning it can be either an empty dict, a dict with values, or an Options BaseModel instance.

2. **Dict iteration yields keys, not pairs**: When Python iterates over a dict, it yields only the keys (strings), not (key, value) tuples. The dict comprehension tries to unpack each key string into two variables `key` and `value`.

3. **Unpacking failure**: For keys with length != 2, Python raises `ValueError: too many values to unpack` or `ValueError: not enough values to unpack`. For keys with exactly 2 characters, it incorrectly unpacks the string itself (e.g., "ab" becomes key='a', value='b').

4. **The function lacks documentation**: There's no docstring or type hints indicating what input type is expected, making this ambiguous behavior particularly problematic.

## Relevant Context

The correct pattern is already used elsewhere in the codebase. For example, at line 874 in models.py:
```python
for key, value in dict(self.prompt.options).items()
```

This shows the developers understand that when working with dict-like objects, `.items()` should be used to get (key, value) pairs.

The Options class inherits from Pydantic's BaseModel, which when iterated does yield (key, value) tuples. However, the Prompt class explicitly allows regular Python dicts as well, making the not_nulls function incompatible with valid API usage.

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,5 +913,8 @@ def combine_chunks(chunks: List) -> dict:


 def not_nulls(data) -> dict:
+    """Filter out None values from a dict or iterable of (key, value) pairs."""
+    if isinstance(data, dict):
+        data = data.items()
     return {key: value for key, value in data if value is not None}
```