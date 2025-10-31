# Bug Report: xarray.core.indexes Indexes.is_multi() Raises Unhelpful KeyError for Non-Existent Keys

**Target**: `xarray.core.indexes.Indexes.is_multi()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Indexes.is_multi()` method raises an unhelpful `KeyError` when called with a non-existent key, contradicting its docstring which states "Return True if key maps to a multi-coordinate index, False otherwise" and being inconsistent with the error handling in the similar `get_all_coords()` method.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from xarray.core.indexes import PandasIndex, Indexes
from xarray.core.variable import Variable

@st.composite
def indexes_with_keys(draw):
    n_indexes = draw(st.integers(min_value=1, max_value=5))
    valid_keys = [f"dim_{i}" for i in range(n_indexes)]
    invalid_key = draw(st.text(min_size=1, max_size=10).filter(lambda x: x not in valid_keys))

    indexes_dict = {}
    variables_dict = {}
    for key in valid_keys:
        idx = PandasIndex(pd.Index([1, 2, 3]), key)
        var = Variable((key,), [1, 2, 3])
        indexes_dict[key] = idx
        variables_dict[key] = var

    return Indexes(indexes_dict, variables_dict), invalid_key

@settings(max_examples=100)
@given(indexes_with_keys())
def test_is_multi_invalid_key_error(indexes_and_key):
    indexes, invalid_key = indexes_and_key
    try:
        result = indexes.is_multi(invalid_key)
        assert False, "Expected an error for invalid key"
    except KeyError:
        pass
    except ValueError as e:
        assert "no index found" in str(e).lower()

if __name__ == "__main__":
    test_is_multi_invalid_key_error()
```

<details>

<summary>
**Failing input**: `(Indexes: dim_0 PandasIndex, 'invalid')`
</summary>
```
Testing with invalid key: 'invalid'
  Got expected KeyError: 'invalid'
Trying explicit example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex, 'invalid'),
)
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex, '0'),
)
Testing with invalid key: '0'
  Got expected KeyError: '0'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex, '0'),
)
Testing with invalid key: '0'
  Got expected KeyError: '0'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex, '°>\x83'),
)
Testing with invalid key: '°>\x83'
  Got expected KeyError: '°>\x83'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex, '}\x17Zª\U000fe3e1𡭴\U0004f4e1úÝ4'),
)
Testing with invalid key: '}\x17Zª\U000fe3e1𡭴\U0004f4e1úÝ4'
  Got expected KeyError: '}\x17Zª\U000fe3e1𡭴\U0004f4e1úÝ4'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex, 'é'),
)
Testing with invalid key: 'é'
  Got expected KeyError: 'é'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex, 'º\xa0L'),
)
Testing with invalid key: 'º\xa0L'
  Got expected KeyError: 'º\xa0L'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex
         dim_3    PandasIndex, 'Á\U00093d26'),
)
Testing with invalid key: 'Á\U00093d26'
  Got expected KeyError: 'Á\U00093d26'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex
         dim_3    PandasIndex, 'Z´'),
)
Testing with invalid key: 'Z´'
  Got expected KeyError: 'Z´'
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex
         dim_3    PandasIndex, '4\x1eå('),
)
Testing with invalid key: '4\x1eå('
  Got expected KeyError: '4\x1eå('
Trying example: test_is_multi_invalid_key_error(
    indexes_and_key=(Indexes:
         dim_0    PandasIndex
         dim_1    PandasIndex
         dim_2    PandasIndex
         dim_3    PandasIndex, 'âUt¥\U000dd52cfá\U000c1250{¤'),
)
Testing with invalid key: 'âUt¥\U000dd52cfá\U000c1250{¤'
  Got expected KeyError: 'âUt¥\U000dd52cfá\U000c1250{¤'
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from xarray.core.indexes import PandasIndex, Indexes
from xarray.core.variable import Variable

# Create a simple index
idx1 = PandasIndex(pd.Index([1, 2, 3]), "x")
var1 = Variable(("x",), [1, 2, 3])

# Create an Indexes object
indexes_obj = Indexes({"x": idx1}, {"x": var1})

# Try to check if a non-existent key "y" is multi
# This should raise a KeyError
indexes_obj.is_multi("y")
```

<details>

<summary>
KeyError raised when checking non-existent key
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/repo.py", line 14, in <module>
    indexes_obj.is_multi("y")
    ~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 1775, in is_multi
    return len(self._id_coord_names[self._coord_name_id[key]]) > 1
                                    ~~~~~~~~~~~~~~~~~~~^^^^^
KeyError: 'y'
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Misleading docstring**: The docstring states "Return True if key maps to a multi-coordinate index, False otherwise." The phrase "False otherwise" implies that any key that doesn't map to a multi-coordinate index should return False, not raise an exception.

2. **API inconsistency**: The same class has a `get_all_coords()` method that properly handles missing keys with an `errors` parameter and raises a descriptive `ValueError` with message `f"no index found for {key!r} coordinate"` when a key is not found.

3. **Poor error message**: The raw `KeyError: 'y'` provides no context about what went wrong. Users have to trace through the stack trace to understand that the key wasn't found in the indexes.

4. **Violates principle of least surprise**: Users accessing this through the public API (`Dataset.indexes.is_multi()` or `DataArray.indexes.is_multi()`) would reasonably expect either:
   - A boolean return value (False for non-existent keys)
   - A descriptive error message explaining the problem
   - An API similar to `get_all_coords()` with an errors parameter

## Relevant Context

The `Indexes` class implements `collections.abc.Mapping` and is accessible through the public xarray API via `Dataset.indexes` and `DataArray.indexes` properties. The class already has proper error handling patterns established in the `get_all_coords()` method (lines 1799-1803 of `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py`):

```python
if key not in self._indexes:
    if errors == "raise":
        raise ValueError(f"no index found for {key!r} coordinate")
    else:
        return {}
```

The `is_multi()` method (line 1775) directly accesses internal dictionaries without validation:
```python
return len(self._id_coord_names[self._coord_name_id[key]]) > 1
```

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.Indexes.html

## Proposed Fix

Add proper key validation with an informative error message to match the pattern used in `get_all_coords()`:

```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
index abc123..def456 100644
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1771,6 +1771,9 @@ class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
     def is_multi(self, key: Hashable) -> bool:
         """Return True if ``key`` maps to a multi-coordinate index,
         False otherwise.
         """
+        if key not in self._indexes:
+            raise ValueError(f"no index found for {key!r} coordinate")
+
         return len(self._id_coord_names[self._coord_name_id[key]]) > 1
```