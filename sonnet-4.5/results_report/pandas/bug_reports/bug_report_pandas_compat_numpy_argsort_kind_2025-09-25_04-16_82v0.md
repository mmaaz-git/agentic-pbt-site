# Bug Report: pandas.compat.numpy validate_argsort_kind Missing 'kind' Parameter

**Target**: `pandas.compat.numpy.function.validate_argsort_kind`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_argsort_kind` validator is missing the 'kind' parameter in its allowed parameters dictionary (ARGSORT_DEFAULTS_KIND), despite its name and code comment indicating it should support this parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from pandas.compat.numpy.function import validate_argsort_with_ascending


@given(ascending=st.booleans(),
       kind=st.one_of(st.none(), st.just('quicksort'), st.just('mergesort')))
def test_validate_argsort_kind_accepts_kind_parameter(ascending, kind):
    kwargs = {"kind": kind} if kind is not None else {}

    try:
        result = validate_argsort_with_ascending(ascending, (), kwargs)
        assert isinstance(result, bool)
    except TypeError as e:
        if "unexpected keyword argument 'kind'" in str(e):
            pytest.fail(f"validate_argsort_kind should accept 'kind' parameter but got: {e}")
        raise
```

**Failing input**: `ascending=True`, `kwargs={"kind": None}`

## Reproducing the Bug

```python
from pandas.compat.numpy.function import (
    ARGSORT_DEFAULTS,
    ARGSORT_DEFAULTS_KIND,
    validate_argsort_with_ascending,
)

print("ARGSORT_DEFAULTS includes 'kind':", 'kind' in ARGSORT_DEFAULTS)
print("ARGSORT_DEFAULTS_KIND includes 'kind':", 'kind' in ARGSORT_DEFAULTS_KIND)

ascending = True
kwargs = {"kind": None}

try:
    result = validate_argsort_with_ascending(ascending, (), kwargs)
    print(f"Success: {result}")
except TypeError as e:
    print(f"Bug: {e}")
```

Output:
```
ARGSORT_DEFAULTS includes 'kind': True
ARGSORT_DEFAULTS_KIND includes 'kind': False
Bug: argsort() got an unexpected keyword argument 'kind'
```

## Why This Is A Bug

1. The code comment states: "two different signatures of argsort, this second validation for when the `kind` param is supported"
2. The name `validate_argsort_kind` implies it should handle the 'kind' parameter
3. ARGSORT_DEFAULTS includes 'kind' (set to None), indicating it should be a valid parameter
4. However, ARGSORT_DEFAULTS_KIND (used by validate_argsort_kind) excludes 'kind', causing TypeError when users pass `kind=None` even though that's the default value
5. This breaks the API contract and makes `validate_argsort_with_ascending` reject valid inputs

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -155,6 +155,7 @@ validate_argsort = CompatValidator(
 # `kind` param is supported
 ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
 ARGSORT_DEFAULTS_KIND["axis"] = -1
+ARGSORT_DEFAULTS_KIND["kind"] = None
 ARGSORT_DEFAULTS_KIND["order"] = None
 ARGSORT_DEFAULTS_KIND["stable"] = None
 validate_argsort_kind = CompatValidator(
```
