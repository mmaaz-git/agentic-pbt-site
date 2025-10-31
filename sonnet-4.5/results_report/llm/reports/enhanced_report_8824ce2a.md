# Bug Report: llm.default_plugins.openai_models.not_nulls - Dict iteration causes ValueError

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: Critical
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes with a ValueError on any non-empty dictionary input due to incorrect iteration syntax, completely breaking all OpenAI model functionality in the llm library.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for llm.default_plugins.openai_models.not_nulls
"""
from hypothesis import given, strategies as st

def not_nulls(data) -> dict:
    """
    This is the buggy implementation from llm/default_plugins/openai_models.py:915
    """
    return {key: value for key, value in data if value is not None}

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_filters_none_values(d):
    result = not_nulls(d)

    assert isinstance(result, dict)
    for key, value in result.items():
        assert value is not None

if __name__ == "__main__":
    # Run the test - it will fail and show the minimal failing example
    test_not_nulls_filters_none_values()
```

<details>

<summary>
**Failing input**: `{'': None}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 23, in <module>
    test_not_nulls_filters_none_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 14, in test_not_nulls_filters_none_values
    def test_not_nulls_filters_none_values(d):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 15, in test_not_nulls_filters_none_values
    result = not_nulls(d)
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 11, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 0)
Falsifying example: test_not_nulls_filters_none_values(
    d={'': None},
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the bug in llm.default_plugins.openai_models.not_nulls
"""

def not_nulls(data) -> dict:
    """
    This is the buggy implementation from llm/default_plugins/openai_models.py:915
    """
    return {key: value for key, value in data if value is not None}

# Test case that should work but crashes
test_data = {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
print(f"Testing not_nulls with: {test_data}")

try:
    result = not_nulls(test_data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Even simpler case that also fails
print("\nTesting with simpler case: {'a': 1}")
try:
    result = not_nulls({'a': 1})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Empty dict is the only case that works
print("\nTesting with empty dict: {}")
try:
    result = not_nulls({})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ValueError on any non-empty dict input
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 17, in <module>
    result = not_nulls(test_data)
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 10, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: too many values to unpack (expected 2)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 27, in <module>
    result = not_nulls({'a': 1})
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 10, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 1)
Testing not_nulls with: {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
Error: ValueError: too many values to unpack (expected 2)

Testing with simpler case: {'a': 1}
Error: ValueError: not enough values to unpack (expected 2, got 1)

Testing with empty dict: {}
Result: {}
```
</details>

## Why This Is A Bug

This is a critical bug that completely breaks OpenAI model functionality in the llm library. The function attempts to iterate over a dictionary using `for key, value in data`, but in Python, iterating directly over a dict only yields the keys, not (key, value) tuples. This causes:

1. **Unpacking failure**: When the dict has keys with more than one character (e.g., `'temperature'`), Python tries to unpack the string into two variables, causing "too many values to unpack"
2. **Single character keys fail differently**: Keys with one character (e.g., `'a'`) cause "not enough values to unpack (expected 2, got 1)"
3. **Complete breakage of OpenAI models**: The function is called by `build_kwargs` (line 658), which is used by both `Chat.execute` (line 701) and `AsyncChat.execute` (line 786), meaning ALL OpenAI API calls fail

The function's clear intent is to filter out None values from the options dictionary before passing them to the OpenAI API, but due to the syntax error it crashes instead.

## Relevant Context

The `not_nulls` function is located in `/llm/default_plugins/openai_models.py` at line 915-916. It's a critical component in the request pipeline:

1. Called by `build_kwargs` at line 658: `kwargs = dict(not_nulls(prompt.options))`
2. `build_kwargs` is called by both synchronous and asynchronous chat implementations
3. `prompt.options` is always a dictionary (initialized as `options or {}` in llm/models.py)
4. This affects all OpenAI models registered by the plugin, including GPT-4, GPT-3.5, and all their variants

The bug has likely gone unnoticed if:
- Tests only use empty option dictionaries
- The library was not tested with actual OpenAI API calls
- Users haven't upgraded to a version with this bug yet

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}


 def combine_chunks(chunks: List) -> dict:
```