# Bug Report: llm.utils.remove_dict_none_values - None Values Remain in Lists

**Target**: `llm.utils.remove_dict_none_values`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_dict_none_values` function fails to remove `None` values from lists, despite recursively processing list elements. This causes `None` values to remain in the output structure, contradicting the function's purpose of cleaning up data structures.

## Property-Based Test

```python
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
```

**Failing input**: `{"a": [None, 1]}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import remove_dict_none_values

input_dict = {"choices": [None, {"text": "response"}]}

result = remove_dict_none_values(input_dict)

print(f"Input:  {input_dict}")
print(f"Output: {result}")
```

Output:
```
Input:  {'choices': [None, {'text': 'response'}]}
Output: {'choices': [None, {'text': 'response'}]}
```

The `None` value in the list is preserved, when it should be filtered out.

## Why This Is A Bug

The function's purpose is to recursively clean up data structures by removing `None` values. This is evident from:

1. **Usage context**: The function is used in `openai_models.py` to clean API responses before storing them: `response.response_json = remove_dict_none_values(completion.model_dump())`

2. **Recursive list processing**: The function explicitly recurses into lists (line 91-92 in utils.py), indicating intent to handle nested structures comprehensively.

3. **Inconsistent behavior**: The function removes `None` values from dict values but not from list elements, creating an inconsistent API.

The current implementation (lines 91-92):
```python
elif isinstance(value, list):
    new_dict[key] = [remove_dict_none_values(v) for v in value]
```

This calls `remove_dict_none_values` on each list element, but since `remove_dict_none_values(None)` returns `None` (line 82-83), the `None` values are preserved.

## Fix

```diff
diff --git a/utils.py b/utils.py
index 1234567..abcdefg 100644
--- a/utils.py
+++ b/utils.py
@@ -89,7 +89,10 @@ def remove_dict_none_values(d):
                 if nested:
                     new_dict[key] = nested
             elif isinstance(value, list):
-                new_dict[key] = [remove_dict_none_values(v) for v in value]
+                # Filter out None values from lists and recursively process remaining elements
+                filtered = [remove_dict_none_values(v) for v in value if v is not None]
+                if filtered:  # Only include non-empty lists
+                    new_dict[key] = filtered
             else:
                 new_dict[key] = value
     return new_dict
```

This fix:
1. Filters out `None` values from lists before processing
2. Only includes the list in the output if it's non-empty after filtering
3. Maintains consistency with how empty dicts are handled (line 89-90)