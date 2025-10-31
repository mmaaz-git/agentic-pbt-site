# Bug Report: llm.default_plugins.openai_models.not_nulls Dictionary Iteration Bug Causes Universal Crash

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes with a `ValueError` on all dictionary inputs due to missing `.items()` in the dictionary comprehension, preventing any OpenAI model from being used with options.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    assert isinstance(result, dict)
    assert all(value is not None for value in result.values())

if __name__ == "__main__":
    test_not_nulls_removes_none_values()
```

<details>

<summary>
**Failing input**: `{'': None}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 11, in <module>
    test_not_nulls_removes_none_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 5, in test_not_nulls_removes_none_values
    def test_not_nulls_removes_none_values(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 6, in test_not_nulls_removes_none_values
    result = not_nulls(data)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/default_plugins/openai_models.py", line 916, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 0)
Falsifying example: test_not_nulls_removes_none_values(
    data={'': None},
)
```
</details>

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import not_nulls

# Test case from the bug report
data = {'': None}
try:
    result = not_nulls(data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Additional test with non-empty keys
data2 = {'key1': 'value1', 'key2': None, 'key3': 42}
try:
    result2 = not_nulls(data2)
    print(f"Result2: {result2}")
except Exception as e:
    print(f"Error2: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError raised for all dictionary inputs
</summary>
```
Error: ValueError: not enough values to unpack (expected 2, got 0)
Error2: ValueError: too many values to unpack (expected 2)
```
</details>

## Why This Is A Bug

The function violates Python's dictionary iteration semantics. When iterating over a dictionary directly with `for key, value in data`, Python attempts to unpack each dictionary key (not key-value pairs) into two variables. Since dictionary iteration yields only keys, not tuples, this causes:

1. **For empty string keys**: `ValueError: not enough values to unpack (expected 2, got 0)` because an empty string has 0 characters
2. **For keys with >2 characters**: `ValueError: too many values to unpack (expected 2)` because the string has more than 2 characters
3. **For keys with exactly 2 characters**: Would incorrectly unpack the two characters as key and value

The function's type signature `def not_nulls(data) -> dict:` and its usage context at line 658 (`kwargs = dict(not_nulls(prompt.options))`) clearly indicate it should accept a dictionary and return a filtered dictionary without None values. This is a fundamental Python programming error that makes the entire OpenAI plugin unusable when options are provided.

## Relevant Context

- **Location**: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py` lines 915-916
- **Usage**: Called in `build_kwargs` method (line 658) as part of the OpenAI model invocation pipeline
- **Impact**: Affects all OpenAI models including GPT-3.5, GPT-4, GPT-5, o1, o3 series when invoked with options
- **Similar correct implementation**: The same file imports and uses `remove_dict_none_values` from `llm/utils.py` which correctly uses `.items()` for dictionary iteration
- **Python documentation**: [Dictionary iteration](https://docs.python.org/3/tutorial/datastructures.html#looping-techniques) clearly shows `.items()` is required for key-value iteration

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,4 +913,4 @@ class Completion(Chat):

 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```