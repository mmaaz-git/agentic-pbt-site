# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Dead Code

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas/compat/numpy/function.py` contains dead code: line 138 sets `kind='quicksort'`, but line 140 immediately overwrites it with `kind=None`. This is confusing and violates code clarity principles.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


def test_argsort_defaults_kind_consistency():
    """
    Property: Dictionary values should not be overwritten without reason.
    The final value should be the only assignment.
    """
    assert 'kind' in ARGSORT_DEFAULTS
    assert ARGSORT_DEFAULTS['kind'] is None
```

**Failing expectation**: When reading the source code, one would expect `kind` to be set only once, but it's set twice with different values.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/pandas')

with open('pandas/compat/numpy/function.py', 'r') as f:
    lines = f.readlines()
    print("Line 138:", lines[137].strip())
    print("Line 140:", lines[139].strip())

from pandas.compat.numpy.function import ARGSORT_DEFAULTS
print(f"Final value: ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")
```

Output:
```
Line 138: ARGSORT_DEFAULTS["kind"] = "quicksort"
Line 140: ARGSORT_DEFAULTS["kind"] = None
Final value: ARGSORT_DEFAULTS['kind'] = None
```

## Why This Is A Bug

The assignment on line 138 serves no purpose and will never affect program behavior because line 140 immediately overwrites it. This is dead code that:

1. Confuses readers about the intended default value
2. Suggests uncertainty or incomplete refactoring during development
3. Wastes a small amount of execution time (negligible but unnecessary)
4. Violates the principle of code clarity

The intended value is `None` (as evidenced by line 140 and the actual behavior), so line 138 should be removed.

## Fix

Remove the dead code assignment on line 138:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,7 +135,6 @@ def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```