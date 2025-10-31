# Bug Report: llm.utils.remove_dict_none_values Doesn't Remove None from Lists

**Target**: `llm.utils.remove_dict_none_values`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_dict_none_values` function preserves `None` values inside lists while removing them from dictionary values, creating an inconsistency in how None values are handled across nested data structures.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for llm.utils.remove_dict_none_values using Hypothesis"""

from llm.utils import remove_dict_none_values
from hypothesis import given, strategies as st

@given(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.one_of(st.none(), st.integers()))
    )
))
def test_remove_dict_none_values_removes_all_nones(d):
    result = remove_dict_none_values(d)

    def has_none(obj):
        if obj is None:
            return True
        if isinstance(obj, dict):
            return any(has_none(v) for v in obj.values())
        if isinstance(obj, list):
            return any(has_none(v) for v in obj)
        return False

    assert not has_none(result), f"Result still contains None: {result}"

# Run the test
if __name__ == "__main__":
    try:
        test_remove_dict_none_values_removes_all_nones()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        print("\nThis confirms the bug - the function does not remove None values from lists.")
```

<details>

<summary>
**Failing input**: `{'0': [None]}`
</summary>
```
Test failed with error: Result still contains None: {'0': [None]}

This confirms the bug - the function does not remove None values from lists.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the bug in llm.utils.remove_dict_none_values"""

from llm.utils import remove_dict_none_values

# Test the failing input reported
test_dict = {"a": [1, None, 3]}
result = remove_dict_none_values(test_dict)

print("Test 1: Simple list with None")
print(f"Input:  {test_dict}")
print(f"Output: {result}")
print(f"None still present: {None in result.get('a', [])}")
print()

# Test with nested structures
test_dict2 = {"a": [1, None, 3], "b": {"c": [None, 2]}}
result2 = remove_dict_none_values(test_dict2)

print("Test 2: Nested dict with lists containing None")
print(f"Input:  {test_dict2}")
print(f"Output: {result2}")
print(f"None in a: {None in result2.get('a', [])}")
print(f"None in b.c: {None in result2.get('b', {}).get('c', [])}")
print()

# Test with None as direct dict value
test_dict3 = {"a": None, "b": [None], "c": {"d": None, "e": [None]}}
result3 = remove_dict_none_values(test_dict3)

print("Test 3: Mix of None in dict values and lists")
print(f"Input:  {test_dict3}")
print(f"Output: {result3}")
print()

# Show what the docstring claims
print("Function docstring:")
print(f'"{remove_dict_none_values.__doc__.strip()}"')
```

<details>

<summary>
Demonstration of None values preserved in lists
</summary>
```
Test 1: Simple list with None
Input:  {'a': [1, None, 3]}
Output: {'a': [1, None, 3]}
None still present: True

Test 2: Nested dict with lists containing None
Input:  {'a': [1, None, 3], 'b': {'c': [None, 2]}}
Output: {'a': [1, None, 3], 'b': {'c': [None, 2]}}
None in a: True
None in b.c: True

Test 3: Mix of None in dict values and lists
Input:  {'a': None, 'b': [None], 'c': {'d': None, 'e': [None]}}
Output: {'b': [None], 'c': {'e': [None]}}

Function docstring:
"Recursively remove keys with value of None or value of a dict that is all values of None"
```
</details>

## Why This Is A Bug

While the docstring states "Recursively remove keys with value of None", the function's behavior is inconsistent:

1. **Inconsistent handling**: The function removes `None` values that are direct dictionary values (e.g., `{"a": None}` becomes `{}`), but preserves `None` values within lists (e.g., `{"a": [None]}` stays as `{"a": [None]}`).

2. **Already processes lists**: Line 92 of the function explicitly handles lists with `[remove_dict_none_values(v) for v in value]`, showing intent to recursively process list contents.

3. **Root cause**: When `remove_dict_none_values(None)` is called (for a None element in a list), it returns `None` unchanged (line 83: `if not isinstance(d, dict): return d`). This None is then included in the list comprehension result.

4. **Docstring ambiguity**: The docstring could be interpreted to mean either:
   - Remove only dictionary keys with None values (narrow interpretation - current behavior)
   - Recursively remove all None values from the data structure (broad interpretation - expected behavior)

## Relevant Context

This function is used internally in the `llm` package's OpenAI models plugin (`/llm/default_plugins/openai_models.py`) to clean up API response data before storage. The function is called multiple times to process OpenAI API responses with `remove_dict_none_values(completion.model_dump())` and similar patterns.

The issue is less critical because:
- The function technically behaves according to a narrow reading of its docstring
- OpenAI API responses may rarely have None values within lists
- It's an internal utility function, not a public API
- Users encountering this can work around it if needed

Source code location: `/lib/python3.13/site-packages/llm/utils.py` lines 78-95
Documentation: No external documentation found; only the function's docstring exists

## Proposed Fix

```diff
def remove_dict_none_values(d):
    """
-   Recursively remove keys with value of None or value of a dict that is all values of None
+   Recursively remove keys with value of None, empty dicts, and None values from lists
    """
    if not isinstance(d, dict):
        return d
    new_dict = {}
    for key, value in d.items():
        if value is not None:
            if isinstance(value, dict):
                nested = remove_dict_none_values(value)
                if nested:
                    new_dict[key] = nested
            elif isinstance(value, list):
-               new_dict[key] = [remove_dict_none_values(v) for v in value]
+               filtered_list = [remove_dict_none_values(v) for v in value if v is not None]
+               if filtered_list:
+                   new_dict[key] = filtered_list
            else:
                new_dict[key] = value
    return new_dict
```