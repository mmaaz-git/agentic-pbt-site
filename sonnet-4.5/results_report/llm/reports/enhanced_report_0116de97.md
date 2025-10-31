# Bug Report: llm.utils.remove_dict_none_values - None Values Persist in Lists

**Target**: `llm.utils.remove_dict_none_values`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_dict_none_values` function fails to remove `None` values from lists while correctly removing them from dictionary values, creating inconsistent behavior that contradicts the function's apparent purpose of cleaning data structures.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.utils import remove_dict_none_values


def has_none_value(obj):
    """Recursively check if obj contains any None values"""
    if obj is None:
        return True
    if isinstance(obj, dict):
        for value in obj.values():
            if has_none_value(value):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if has_none_value(item):
                return True
    return False


@given(st.recursive(
    st.one_of(st.none(), st.integers(), st.text(), st.booleans()),
    lambda children: st.one_of(
        st.dictionaries(st.text(), children, max_size=5),
        st.lists(children, max_size=5)
    ),
    max_leaves=10
))
@settings(max_examples=1000)
def test_remove_dict_none_values_removes_all_none(d):
    result = remove_dict_none_values(d)
    if isinstance(result, dict):
        assert not has_none_value(result), \
            f"Found None values in output:\nInput: {d}\nOutput: {result}"


if __name__ == "__main__":
    test_remove_dict_none_values_removes_all_none()
```

<details>

<summary>
**Failing input**: `{'': [None]}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 40, in <module>
    test_remove_dict_none_values_removes_all_none()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 24, in test_remove_dict_none_values_removes_all_none
    st.one_of(st.none(), st.integers(), st.text(), st.booleans()),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 35, in test_remove_dict_none_values_removes_all_none
    assert not has_none_value(result), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Found None values in output:
Input: {'': [None]}
Output: {'': [None]}
Falsifying example: test_remove_dict_none_values_removes_all_none(
    d={'': [None]},
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/6/hypo.py:11
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import remove_dict_none_values

# Test case from the bug report
input_dict = {"choices": [None, {"text": "response"}]}

result = remove_dict_none_values(input_dict)

print(f"Input:  {input_dict}")
print(f"Output: {result}")
print()

# Additional test cases to demonstrate the inconsistency
test_cases = [
    {"a": [None, 1]},
    {"b": [None]},
    {"c": [1, None, 2]},
    {"d": {"nested": None}},  # This should remove the None
    {"e": [{"nested": None}]},  # What happens here?
    {"": [None]},  # Minimal failing case
]

print("Additional test cases:")
print("=" * 50)
for test in test_cases:
    result = remove_dict_none_values(test)
    print(f"Input:  {test}")
    print(f"Output: {result}")
    print()
```

<details>

<summary>
None values remain in lists but are removed from dicts
</summary>
```
Input:  {'choices': [None, {'text': 'response'}]}
Output: {'choices': [None, {'text': 'response'}]}

Additional test cases:
==================================================
Input:  {'a': [None, 1]}
Output: {'a': [None, 1]}

Input:  {'b': [None]}
Output: {'b': [None]}

Input:  {'c': [1, None, 2]}
Output: {'c': [1, None, 2]}

Input:  {'d': {'nested': None}}
Output: {}

Input:  {'e': [{'nested': None}]}
Output: {'e': [{}]}

Input:  {'': [None]}
Output: {'': [None]}

```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Function Name Implies Complete Removal**: The function is named `remove_dict_none_values`, which strongly suggests it should remove ALL None values from the data structure, not just those at the dictionary level.

2. **Inconsistent Behavior**: The function exhibits contradictory behavior:
   - It REMOVES None values from dictionary values (line 86: `if value is not None`)
   - It PRESERVES None values in lists (lines 91-92)
   - Example: `{"d": {"nested": None}}` returns `{}` (None removed)
   - But `{"a": [None, 1]}` returns `{"a": [None, 1]}` (None preserved)

3. **Documentation Ambiguity**: The docstring states "Recursively remove keys with value of None or value of a dict that is all values of None" but doesn't explicitly mention list handling. However, the function actively processes lists (lines 91-92), showing intent to handle them comprehensively.

4. **Recursive Processing Incomplete**: The function calls `remove_dict_none_values(v) for v in value` on list elements (line 92), but since `remove_dict_none_values(None)` returns `None` (lines 82-83), the None values are reinserted into the list.

5. **Real-World Impact**: The function is used extensively in `openai_models.py` to clean API responses before storage. API responses often contain lists with None values that should be cleaned up for consistent data processing.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py` lines 78-95.

Usage in the codebase shows it's primarily used for cleaning OpenAI API responses:
- `openai_models.py`: Used 6 times to clean `completion.model_dump()` and chunk combinations
- Called with `response.response_json = remove_dict_none_values(completion.model_dump())`

The current implementation creates a situation where developers must manually filter None values from lists after calling this function, defeating its purpose as a comprehensive cleaning utility.

GitHub repository: The `llm` package appears to be Simon Willison's LLM tool for interacting with Large Language Models. This bug affects data cleaning operations that are crucial for proper API response handling.

## Proposed Fix

```diff
--- a/utils.py
+++ b/utils.py
@@ -89,7 +89,11 @@ def remove_dict_none_values(d):
                 if nested:
                     new_dict[key] = nested
             elif isinstance(value, list):
-                new_dict[key] = [remove_dict_none_values(v) for v in value]
+                # Filter out None values from lists and recursively process remaining elements
+                filtered = [remove_dict_none_values(v) for v in value if v is not None]
+                # Only include the list if it's non-empty after filtering
+                if filtered:
+                    new_dict[key] = filtered
             else:
                 new_dict[key] = value
     return new_dict
```