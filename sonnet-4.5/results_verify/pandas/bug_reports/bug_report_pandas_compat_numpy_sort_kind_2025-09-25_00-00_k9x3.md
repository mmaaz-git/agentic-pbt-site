# Bug Report: pandas.compat.numpy SORT_DEFAULTS Kind Mismatch

**Target**: `pandas.compat.numpy.function.SORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

SORT_DEFAULTS['kind'] is set to 'quicksort', but numpy.sort's actual default for the 'kind' parameter is None. This causes validate_sort to reject numpy's actual default value (kind=None) while accepting 'quicksort', which is incorrect validation behavior.

## Property-Based Test

```python
import inspect
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from pandas.compat.numpy.function import validate_sort, SORT_DEFAULTS


def test_sort_defaults_should_match_numpy():
    """
    Property: SORT_DEFAULTS should contain numpy.sort's actual default values
    so that validation correctly accepts numpy defaults.
    """
    sig = inspect.signature(np.sort)
    numpy_kind_default = sig.parameters['kind'].default

    assert SORT_DEFAULTS['kind'] == numpy_kind_default, \
        f"SORT_DEFAULTS['kind']={SORT_DEFAULTS['kind']!r} " \
        f"but numpy default is {numpy_kind_default!r}"


def test_validate_sort_should_accept_numpy_default():
    """
    Property: validate_sort should accept numpy.sort's actual default value.
    """
    validate_sort((), {'kind': None})
```

**Failing input**: `{'kind': None}`

## Reproducing the Bug

```python
import inspect
import numpy as np
from pandas.compat.numpy.function import validate_sort, SORT_DEFAULTS

sig = inspect.signature(np.sort)
numpy_default = sig.parameters['kind'].default

print(f"numpy.sort default for 'kind': {numpy_default!r}")
print(f"SORT_DEFAULTS['kind']: {SORT_DEFAULTS['kind']!r}")

validate_sort((), {'kind': None})
```

**Output**:
```
numpy.sort default for 'kind': None
SORT_DEFAULTS['kind']: 'quicksort'
ValueError: the 'kind' parameter is not supported in the pandas implementation of sort()
```

## Why This Is A Bug

The purpose of the SORT_DEFAULTS dictionary (and the compat validators in general) is to track numpy's default parameter values so that pandas can validate that users are only passing default values when calling pandas methods via numpy's interface.

When SORT_DEFAULTS['kind'] doesn't match numpy's actual default, the validator incorrectly rejects valid usage (passing kind=None, which is numpy's actual default) and incorrectly accepts invalid usage (passing kind='quicksort', which is not numpy's default in modern versions).

This violates the module's stated purpose from the docstring: "This module provides a set of commonly used default arguments for functions and methods... This module will make it easier to adjust to future upstream changes in the analogous numpy signatures."

**Impact**: grep search of the pandas codebase shows that validate_sort and SORT_DEFAULTS are not currently used anywhere, making this dead/unused code. The bug exists but has no current impact on pandas functionality.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -276,7 +276,7 @@ validate_round = CompatValidator(

 SORT_DEFAULTS: dict[str, int | str | None] = {}
 SORT_DEFAULTS["axis"] = -1
-SORT_DEFAULTS["kind"] = "quicksort"
+SORT_DEFAULTS["kind"] = None
 SORT_DEFAULTS["order"] = None
 validate_sort = CompatValidator(SORT_DEFAULTS, fname="sort", method="kwargs")
```