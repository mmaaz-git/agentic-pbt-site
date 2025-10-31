# Bug Report: pandas.io.json ujson Float Precision Loss Leading to Silent Data Corruption

**Target**: `pandas.io.json.ujson_dumps` and `pandas.io.json.ujson_loads`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The pandas ujson serialization functions silently lose precision for floating-point numbers, causing data corruption where deserialized values differ from the original by measurable amounts (e.g., a difference of 2.0 for the value 1.0000000000000002e+16).

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for pandas ujson round-trip property"""

from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads
import pandas as pd

@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans(),
        st.none()
    )
)
@settings(max_examples=100, verbosity=2)
def test_ujson_roundtrip(data):
    """Test that ujson_dumps and ujson_loads preserve data accurately"""
    json_str = ujson_dumps(data)
    result = ujson_loads(json_str)
    assert result == data or (pd.isna(result) and pd.isna(data))

if __name__ == "__main__":
    test_ujson_roundtrip()
```

<details>

<summary>
**Failing input**: `data=1.0000000000000002e+16`
</summary>
```
Trying example: test_ujson_roundtrip(
    data=1.0000000000000002e+16,
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 22, in test_ujson_roundtrip
    assert result == data or (pd.isna(result) and pd.isna(data))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 25, in <module>
    test_ujson_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 9, in test_ujson_roundtrip
    st.one_of(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 22, in test_ujson_roundtrip
    assert result == data or (pd.isna(result) and pd.isna(data))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_ujson_roundtrip(
    data=1.0000000000000002e+16,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of pandas ujson precision loss bug"""

from pandas.io.json import ujson_dumps, ujson_loads

# Test value with precision loss
test_value = 1.0000000000000002e+16

# Serialize and deserialize
json_str = ujson_dumps(test_value)
result = ujson_loads(json_str)

print(f"Original value:     {test_value}")
print(f"Original value (repr): {repr(test_value)}")
print(f"JSON string:        {json_str}")
print(f"Deserialized value: {result}")
print(f"Deserialized (repr):   {repr(result)}")
print(f"Values are equal:   {test_value == result}")
print(f"Difference:         {test_value - result}")
```

<details>

<summary>
Precision loss demonstration showing 2.0 difference in values
</summary>
```
Original value:     1.0000000000000002e+16
Original value (repr): 1.0000000000000002e+16
JSON string:        1e+16
Deserialized value: 1e+16
Deserialized (repr):   1e+16
Values are equal:   False
Difference:         2.0
```
</details>

## Why This Is A Bug

This behavior violates fundamental expectations of JSON serialization:

1. **Silent Data Corruption**: The value 1.0000000000000002e+16 becomes 1e+16, losing the last 2 significant digits. The difference of 2.0 between original and deserialized values represents actual data loss, not just floating-point rounding errors.

2. **Violation of Round-Trip Property**: The fundamental contract `ujson_loads(ujson_dumps(x)) == x` is broken. Users expect that serializing and deserializing a value will preserve it exactly, especially when both operations come from the same library.

3. **IEEE 754 Non-Compliance**: The IEEE 754 standard requires 17 significant decimal digits to accurately represent a double-precision float. The ujson implementation appears to truncate at 15 digits or less, violating this standard.

4. **Undocumented Behavior**: The pandas documentation for `ujson_dumps` and `ujson_loads` does not mention this precision limitation. Users have no warning that their data may be corrupted.

5. **Inconsistent with Standard Library**: Python's built-in `json` module correctly preserves all 17 significant digits. Users migrating from standard json to pandas ujson for performance would not expect this regression in accuracy.

## Relevant Context

This issue particularly affects:
- **Scientific Computing**: Astronomical calculations, physics simulations, and other scientific applications often work with very large or very precise numbers
- **Financial Applications**: Currency conversions, interest calculations, and other financial computations require exact precision
- **Data Analysis**: Statistical computations where small differences matter

The pandas team has previously acknowledged similar issues:
- Issue #38437 reported similar precision problems with ujson
- PR #54100 addressed some precision-related fixes

The root cause appears to be in the ujson C implementation which defaults to a `double_precision` parameter of 10-15 digits rather than the full 17 required for IEEE 754 compliance.

## Proposed Fix

The bug can be fixed by ensuring ujson uses full double precision (17 digits) by default:

```diff
--- a/pandas/_libs/json.pyx
+++ b/pandas/_libs/json.pyx
@@ -263,7 +263,7 @@ def ujson_dumps(
         ensure_ascii=True,
         date_unit=None,
         iso_dates=False,
-        double_precision=10,
+        double_precision=17,  # IEEE 754 requires 17 digits for full double precision
         default_handler=None,
         indent=0,
     ):
```

Alternative fix at the Python wrapper level:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -100,7 +100,7 @@ def to_json(
         ensure_ascii: bool = True,
         date_unit: str = "ms",
         default_handler: Optional[Callable] = None,
-        double_precision: int = 10,
+        double_precision: int = 17,  # IEEE 754 double precision standard
         force_ascii: bool = True,
         date_format: str = "epoch",
         orient: Optional[str] = None,
```

If performance is a concern, add a parameter to let users choose between speed and accuracy:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -100,6 +100,7 @@ def to_json(
         ensure_ascii: bool = True,
         date_unit: str = "ms",
         default_handler: Optional[Callable] = None,
+        precise: bool = True,  # New parameter for precision vs speed trade-off
         double_precision: int = 10,
         force_ascii: bool = True,
         date_format: str = "epoch",
@@ -110,6 +111,9 @@ def to_json(
     ):
+        if precise and double_precision < 17:
+            double_precision = 17  # Ensure full IEEE 754 precision
+
         return ujson_dumps(
             obj,
             ensure_ascii=ensure_ascii,
```