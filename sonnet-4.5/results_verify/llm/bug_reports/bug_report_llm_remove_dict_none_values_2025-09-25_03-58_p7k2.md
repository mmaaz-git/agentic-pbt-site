# Bug Report: llm.utils.remove_dict_none_values Doesn't Remove None from Lists

**Target**: `llm.utils.remove_dict_none_values`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_dict_none_values` function's docstring claims it "Recursively remove keys with value of None", but it fails to remove `None` values that appear inside lists.

## Property-Based Test

```python
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
```

**Failing input**: `{"a": [1, None, 3]}`

## Reproducing the Bug

```python
from llm.utils import remove_dict_none_values

test_dict = {"a": [1, None, 3], "b": {"c": [None, 2]}}
result = remove_dict_none_values(test_dict)

print(f"Input:  {test_dict}")
print(f"Output: {result}")
```

Output:
```
Input:  {'a': [1, None, 3], 'b': {'c': [None, 2]}}
Output: {'a': [1, None, 3], 'b': {'c': [None, 2]}}
```

The `None` values inside the lists are not removed!

## Why This Is A Bug

The function's docstring and name suggest it should remove all `None` values recursively. However, it only removes `None` values from dictionary keys, not from list elements.

**Root cause**: Line 92 does `new_dict[key] = [remove_dict_none_values(v) for v in value]`. Since `remove_dict_none_values(None)` returns `None` (line 83: `if not isinstance(d, dict): return d`), the `None` values are preserved in the list comprehension.

This could cause issues when:
1. Users expect all `None` values to be removed before serialization
2. Downstream code doesn't expect `None` in lists
3. The data structure represents API responses where `None` should be omitted

## Fix

```diff
def remove_dict_none_values(d):
    """
    Recursively remove keys with value of None or value of a dict that is all values of None
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