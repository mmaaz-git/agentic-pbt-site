# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS_KIND Missing 'kind' Parameter

**Target**: `pandas.compat.numpy.function.validate_argsort_kind`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_argsort_kind` validator is documented as being used "when the `kind` param is supported" but its defaults dictionary `ARGSORT_DEFAULTS_KIND` is missing the 'kind' key, causing it to incorrectly reject valid `kind` parameter values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.compat.numpy.function import (
    ARGSORT_DEFAULTS_KIND,
    validate_argsort_kind,
)


@settings(max_examples=100)
@given(st.sampled_from(['quicksort', 'mergesort', 'heapsort', 'stable']))
def test_validate_argsort_kind_should_accept_kind_parameter(kind_value):
    try:
        validate_argsort_kind((), {'kind': kind_value})
        assert True
    except TypeError as e:
        if "unexpected keyword argument 'kind'" in str(e):
            raise AssertionError(
                f"validate_argsort_kind rejects 'kind' parameter but is supposed to support it. "
                f"ARGSORT_DEFAULTS_KIND = {ARGSORT_DEFAULTS_KIND}"
            )
        raise
```

**Failing input**: `kind_value='quicksort'`

## Reproducing the Bug

```python
from pandas.compat.numpy.function import validate_argsort_kind, ARGSORT_DEFAULTS_KIND

print("ARGSORT_DEFAULTS_KIND:", ARGSORT_DEFAULTS_KIND)
print("Has 'kind' key:", 'kind' in ARGSORT_DEFAULTS_KIND)

try:
    validate_argsort_kind((), {'kind': 'quicksort'})
    print("SUCCESS: Accepted kind parameter")
except TypeError as e:
    print(f"BUG: {e}")
```

Output:
```
ARGSORT_DEFAULTS_KIND: {'axis': -1, 'order': None, 'stable': None}
Has 'kind' key: False
BUG: argsort() got an unexpected keyword argument 'kind'
```

## Why This Is A Bug

In `pandas/compat/numpy/function.py` lines 148-156, the code comment explicitly states:

```python
# two different signatures of argsort, this second validation for when the
# `kind` param is supported
ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
ARGSORT_DEFAULTS_KIND["axis"] = -1
ARGSORT_DEFAULTS_KIND["order"] = None
ARGSORT_DEFAULTS_KIND["stable"] = None
```

The validator is supposed to support the `kind` parameter according to its name and documentation comment, but it rejects it because `ARGSORT_DEFAULTS_KIND` doesn't include `'kind'` in its defaults. This means any code attempting to use `validate_argsort_kind` to validate argsort calls with the `kind` parameter will fail.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -150,6 +150,7 @@ validate_argsort = CompatValidator(
 ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
 ARGSORT_DEFAULTS_KIND["axis"] = -1
 ARGSORT_DEFAULTS_KIND["order"] = None
+ARGSORT_DEFAULTS_KIND["kind"] = "quicksort"
 ARGSORT_DEFAULTS_KIND["stable"] = None
 validate_argsort_kind = CompatValidator(
     ARGSORT_DEFAULTS_KIND, fname="argsort", max_fname_arg_count=0, method="both"
```