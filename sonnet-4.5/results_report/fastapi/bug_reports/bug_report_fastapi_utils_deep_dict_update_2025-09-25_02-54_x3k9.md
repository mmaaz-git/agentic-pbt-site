# Bug Report: fastapi.utils.deep_dict_update Idempotence Violation

**Target**: `fastapi.utils.deep_dict_update`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deep_dict_update` function violates idempotence when merging dictionaries containing list values. Calling the function multiple times with the same arguments produces different results each time, as lists are concatenated rather than merged idempotently.

## Property-Based Test

```python
import copy
from hypothesis import given, strategies as st

from fastapi.utils import deep_dict_update

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=3
    )
)
def test_deep_dict_update_idempotence_with_lists(update_dict):
    main_dict = {}

    deep_dict_update(main_dict, update_dict)
    first_result = copy.deepcopy(main_dict)

    deep_dict_update(main_dict, update_dict)
    second_result = copy.deepcopy(main_dict)

    assert first_result == second_result, (
        f"Idempotence violated: calling deep_dict_update twice with the same "
        f"update_dict produces different results. First: {first_result}, "
        f"Second: {second_result}"
    )
```

**Failing input**: `{"items": [1, 2, 3]}`

## Reproducing the Bug

```python
from fastapi.utils import deep_dict_update

main = {"items": [1, 2]}
update = {"items": [3]}

deep_dict_update(main, update)
print(main)

deep_dict_update(main, update)
print(main)
```

**Expected output:**
```
{'items': [1, 2, 3]}
{'items': [1, 2, 3]}
```

**Actual output:**
```
{'items': [1, 2, 3]}
{'items': [1, 2, 3, 3]}
```

## Why This Is A Bug

1. **Violates idempotence**: The fundamental property `f(f(x, y), y) = f(x, y)` does not hold. Applying the same update multiple times should not change the result after the first application.

2. **Inconsistent behavior**: The function handles dicts idempotently (deep merging) and simple values idempotently (overwriting), but lists non-idempotently (concatenating). This inconsistency is confusing and error-prone.

3. **Common Python operations are idempotent**: Standard update operations like `dict.update()`, `set.update()`, etc. are all idempotent. Users would reasonably expect similar behavior.

4. **Can cause subtle bugs**: If this function is called multiple times (e.g., in a retry scenario, or when processing multiple configuration sources), it will accumulate duplicate list items, leading to incorrect application state.

## Fix

The bug occurs in the list handling branch at line 195-200 of `fastapi/utils.py`. The current implementation concatenates lists:

```python
elif (
    key in main_dict
    and isinstance(main_dict[key], list)
    and isinstance(update_dict[key], list)
):
    main_dict[key] = main_dict[key] + update_dict[key]
```

**Option 1: Make lists idempotent by converting to sets (if order doesn't matter)**

```diff
 elif (
     key in main_dict
     and isinstance(main_dict[key], list)
     and isinstance(update_dict[key], list)
 ):
-    main_dict[key] = main_dict[key] + update_dict[key]
+    seen = set(main_dict[key])
+    main_dict[key] = main_dict[key] + [x for x in update_dict[key] if x not in seen]
```

**Option 2: Replace lists instead of merging (simpler, matches simple value behavior)**

```diff
 elif (
     key in main_dict
     and isinstance(main_dict[key], list)
     and isinstance(update_dict[key], list)
 ):
-    main_dict[key] = main_dict[key] + update_dict[key]
+    pass  # Fall through to the else branch to replace
```

**Option 3: Document the non-idempotent behavior and rename the function**

If concatenation is intentional, the function should be renamed to `deep_dict_accumulate` or similar, and the behavior should be clearly documented. However, this is not recommended as it violates the principle of least surprise.

**Recommendation**: Option 1 is preferred if the intent is to merge lists without duplicates. Option 2 is simpler and more consistent with how simple values are handled.