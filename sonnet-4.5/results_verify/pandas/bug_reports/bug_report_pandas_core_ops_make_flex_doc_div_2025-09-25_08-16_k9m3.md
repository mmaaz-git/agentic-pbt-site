# Bug Report: pandas.core.ops.make_flex_doc Missing 'div' Support

**Target**: `pandas.core.ops.make_flex_doc`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`make_flex_doc()` raises `KeyError` for the operation name 'div', despite 'div' being a documented flexible wrapper method on both Series and DataFrame alongside other operations like 'add', 'sub', 'mul', 'truediv', etc.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.core.ops as ops


@settings(max_examples=200)
@given(
    op_name=st.sampled_from(['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow']),
    typ=st.sampled_from(['series', 'dataframe']),
)
def test_make_flex_doc_supports_all_flex_methods(op_name, typ):
    result = ops.make_flex_doc(op_name, typ)
    assert isinstance(result, str)
    assert len(result) > 0
```

**Failing input**: `op_name='div'`, `typ='series'` (or 'dataframe')

## Reproducing the Bug

```python
import pandas as pd
import pandas.core.ops as ops

print("Series.div exists:", hasattr(pd.Series, 'div'))
print("DataFrame.div exists:", hasattr(pd.DataFrame, 'div'))

print("\nDataFrame.div is listed in DataFrame docs:")
print("'div' in pd.DataFrame.div.__doc__:", 'div' in pd.DataFrame.div.__doc__)

print("\nmake_flex_doc supports all other flex methods:")
for method in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow']:
    try:
        ops.make_flex_doc(method, 'series')
        print(f"  {method}: ✓")
    except KeyError:
        print(f"  {method}: ✗ KeyError")

print("\nmake_flex_doc fails for 'div':")
try:
    ops.make_flex_doc('div', 'series')
    print("  div: ✓")
except KeyError as e:
    print(f"  div: ✗ KeyError: {e}")
```

## Why This Is A Bug

This violates the principle of least surprise and internal consistency:

1. **Documented method**: DataFrame.div's own docstring explicitly lists `div` among flexible wrappers: "Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`)"
2. **API inconsistency**: All other flex methods (add, sub, mul, etc.) work with make_flex_doc
3. **Contract violation**: The function should handle all documented flex operations uniformly

While 'div' is effectively an alias for 'truediv', it is still a distinct public API method that exists alongside its counterpart, and should be handled consistently with other operations.

## Fix

Add 'div' entry to `_op_descriptions` dictionary in `docstrings.py`. Since 'div' is an alias for 'truediv', the entry should mirror the truediv configuration:

```diff
--- a/pandas/core/ops/docstrings.py
+++ b/pandas/core/ops/docstrings.py
@@ -xxx,6 +xxx,7 @@ _op_descriptions = {
     "mul": {...},
     "truediv": {...},
+    "div": {...},  # Alias for truediv, same behavior
     "floordiv": {...},
     ...
 }
```

The 'div' entry should use the same configuration as 'truediv' with appropriate adjustments to mention it's an alias.