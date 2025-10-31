# Bug Report: pandas.core.interchange categorical_column_to_series Modulo Silently Corrupts Data

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `categorical_column_to_series` function uses modulo arithmetic (`codes % len(categories)`) to index into categories, which silently remaps out-of-bounds categorical codes instead of raising an error. This causes data corruption when the interchange protocol provides invalid codes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np


@given(
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10, unique=True),
    invalid_codes=st.lists(st.integers(min_value=100, max_value=1000), min_size=1, max_size=5)
)
@settings(max_examples=100)
def test_categorical_out_of_bounds_codes_should_fail(categories, invalid_codes):
    from pandas.core.interchange.from_dataframe import categorical_column_to_series
    from pandas.core.interchange.buffer import PandasBuffer
    from pandas.core.interchange.dataframe_protocol import DtypeKind, ColumnNullType
    from pandas.core.interchange.utils import ArrowCTypes, Endianness
    import pandas as pd

    codes = np.array(invalid_codes, dtype=np.int64)

    class MockColumn:
        def __init__(self, codes, categories):
            self.codes = codes
            self.categories_list = categories
            self.offset = 0

        def size(self):
            return len(self.codes)

        @property
        def describe_categorical(self):
            cat_series = pd.Series(self.categories_list)

            class CatColumn:
                def __init__(self, series):
                    self._col = np.array(series)

            return {
                "is_ordered": False,
                "is_dictionary": True,
                "categories": CatColumn(cat_series)
            }

        @property
        def describe_null(self):
            return (ColumnNullType.USE_SENTINEL, -1)

        def get_buffers(self):
            buffer = PandasBuffer(self.codes)
            dtype = (DtypeKind.INT, 64, ArrowCTypes.INT64, Endianness.NATIVE)
            return {
                "data": (buffer, dtype),
                "validity": None,
                "offsets": None
            }

    col = MockColumn(codes, categories)
    result_series, _ = categorical_column_to_series(col)

    for code, value in zip(codes, result_series):
        actual_index = code % len(categories)
        expected_value = categories[actual_index]
        assert value == expected_value, f"Code {code} silently mapped to {value} instead of raising error"
```

**Failing input**: Any categorical column with codes >= len(categories)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.interchange.from_dataframe import categorical_column_to_series
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import DtypeKind, ColumnNullType
from pandas.core.interchange.utils import ArrowCTypes, Endianness
import pandas as pd


categories = ['apple', 'banana', 'cherry']
invalid_codes = np.array([0, 1, 100, 200], dtype=np.int64)


class MockColumn:
    def __init__(self, codes, categories):
        self.codes = codes
        self.categories_list = categories
        self.offset = 0

    def size(self):
        return len(self.codes)

    @property
    def describe_categorical(self):
        cat_series = pd.Series(self.categories_list)

        class CatColumn:
            def __init__(self, series):
                self._col = np.array(series)

        return {
            "is_ordered": False,
            "is_dictionary": True,
            "categories": CatColumn(cat_series)
        }

    @property
    def describe_null(self):
        return (ColumnNullType.USE_SENTINEL, -1)

    def get_buffers(self):
        buffer = PandasBuffer(self.codes)
        dtype = (DtypeKind.INT, 64, ArrowCTypes.INT64, Endianness.NATIVE)
        return {
            "data": (buffer, dtype),
            "validity": None,
            "offsets": None
        }


col = MockColumn(invalid_codes, categories)
result_series, _ = categorical_column_to_series(col)

print(f"Categories: {categories}")
print(f"Codes: {invalid_codes}")
print(f"Result: {result_series.tolist()}")
print(f"\nExpected: Error for codes 100 and 200 (out of bounds)")
print(f"Actual: codes silently mapped:")
print(f"  100 % 3 = 1 → 'banana'")
print(f"  200 % 3 = 2 → 'cherry'")
print(f"\nThis is DATA CORRUPTION!")
```

## Why This Is A Bug

The modulo operation at lines 253-254 in `from_dataframe.py` silently remaps any out-of-bounds categorical code to a valid category index. This violates the expected behavior:

1. **Expected**: Out-of-bounds codes should either:
   - Raise a `ValueError` indicating invalid data
   - Be treated as null/missing values (if code == sentinel value)

2. **Actual**: Any code is silently remapped via modulo arithmetic:
   - Code 100 with 3 categories → 100 % 3 = 1 → categories[1]
   - Code 200 with 3 categories → 200 % 3 = 2 → categories[2]

This causes **silent data corruption** where invalid categorical data gets incorrectly mapped to valid categories, potentially leading to wrong analysis results.

The interchange protocol doesn't guarantee that codes are in-bounds, so pandas must validate them. Pandas' own `Categorical` constructor validates this, but the modulo trick bypasses that validation.

## Fix

Remove the modulo operation and let invalid codes be detected naturally, or add explicit validation:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -250,10 +250,15 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    null_kind, sentinel_val = col.describe_null
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Replace sentinel null values with -1 (pandas null for categoricals)
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            codes = np.where(codes == sentinel_val, -1, codes)
+        # Validate that codes are in valid range
+        if np.any((codes != -1) & ((codes < 0) | (codes >= len(categories)))):
+            raise ValueError(f"Categorical codes must be in range [0, {len(categories)}) or equal to null sentinel")
+        values = categories[codes]
     else:
         values = codes
```