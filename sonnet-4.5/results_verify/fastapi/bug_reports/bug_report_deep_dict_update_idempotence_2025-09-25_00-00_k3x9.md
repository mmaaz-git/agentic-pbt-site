# Bug Report: deep_dict_update Non-Idempotent List Handling

**Target**: `fastapi.openapi.utils.deep_dict_update`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deep_dict_update` function exhibits inconsistent idempotence behavior: nested dictionaries are merged idempotently, but lists are concatenated non-idempotently, causing duplicate entries when the same update is applied multiple times.

## Property-Based Test

```python
import copy
from hypothesis import given, strategies as st
from fastapi.openapi.utils import deep_dict_update


def nested_dict_strategy(max_depth=3):
    if max_depth == 0:
        return st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text(), st.booleans(), st.none()),
            max_size=5
        )

    return st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.text(),
            st.booleans(),
            st.none(),
            st.lists(st.integers(), max_size=3),
            st.deferred(lambda: nested_dict_strategy(max_depth - 1))
        ),
        max_size=3
    )


@given(nested_dict_strategy(), nested_dict_strategy())
def test_deep_dict_update_idempotence(main, update):
    """Applying the same update twice should not change the result after first application"""
    main_copy1 = copy.deepcopy(main)
    main_copy2 = copy.deepcopy(main)

    deep_dict_update(main_copy1, update)
    result_after_first = copy.deepcopy(main_copy1)

    deep_dict_update(main_copy2, update)
    deep_dict_update(main_copy2, update)

    assert main_copy2 == result_after_first
```

**Failing input**: `main={}, update={'0': [0]}`

## Reproducing the Bug

```python
from fastapi.openapi.utils import deep_dict_update

main = {"tags": ["api"]}
update = {"tags": ["v1"]}

deep_dict_update(main, update)
print(main)

deep_dict_update(main, update)
print(main)

deep_dict_update(main, update)
print(main)
```

**Output:**
```
{'tags': ['api', 'v1']}
{'tags': ['api', 'v1', 'v1']}
{'tags': ['api', 'v1', 'v1', 'v1']}
```

## Why This Is A Bug

The function exhibits inconsistent behavior across data types:
- **Dicts**: Merged recursively (idempotent - applying the same merge twice has no additional effect)
- **Lists**: Concatenated (non-idempotent - each application adds duplicate entries)
- **Scalars**: Overwritten (idempotent - applying the same value twice has no additional effect)

This inconsistency violates the principle of least surprise and can lead to bugs in real-world usage. For example, if FastAPI's OpenAPI schema generation is called multiple times (during hot reloading, testing, or error recovery), list fields like `tags`, `security`, or `servers` would accumulate duplicate entries.

## Fix

The list handling should be changed to be idempotent, either by:
1. Extending the list only with items not already present (set union behavior)
2. Replacing the list entirely (consistent with scalar behavior)

Option 2 is simpler and more consistent with the scalar overwrite behavior:

```diff
--- a/fastapi/utils.py
+++ b/fastapi/utils.py
@@ -8,11 +8,6 @@ def deep_dict_update(main_dict: Dict[Any, Any], update_dict: Dict[Any, Any]) ->
             and isinstance(value, dict)
         ):
             deep_dict_update(main_dict[key], value)
-        elif (
-            key in main_dict
-            and isinstance(main_dict[key], list)
-            and isinstance(update_dict[key], list)
-        ):
-            main_dict[key] = main_dict[key] + update_dict[key]
         else:
             main_dict[key] = value
```

This makes lists behave like scalars (overwrite), which is idempotent and consistent with the rest of the function.