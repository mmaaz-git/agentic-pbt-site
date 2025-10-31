# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary contains a duplicate assignment to the `"kind"` key, where it is first correctly set to `"quicksort"` on line 138 and then immediately overwritten with `None` on line 140, causing the parameter validator to incorrectly reject valid values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.data())
def test_argsort_defaults_kind_should_not_be_duplicated(data):
    assert "kind" in ARGSORT_DEFAULTS
    kind_value = ARGSORT_DEFAULTS["kind"]

    assert kind_value is not None, (
        f"ARGSORT_DEFAULTS['kind'] should have a default value, "
        f"but got None. This appears to be due to duplicate assignment "
        f"where 'kind' is first set to 'quicksort' then overwritten to None."
    )

if __name__ == "__main__":
    test_argsort_defaults_kind_should_not_be_duplicated()
```

<details>

<summary>
**Failing input**: `data=data(...)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 20, in <module>
    test_argsort_defaults_kind_should_not_be_duplicated()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 9, in test_argsort_defaults_kind_should_not_be_duplicated
    def test_argsort_defaults_kind_should_not_be_duplicated(data):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 13, in test_argsort_defaults_kind_should_not_be_duplicated
    assert kind_value is not None, (
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: ARGSORT_DEFAULTS['kind'] should have a default value, but got None. This appears to be due to duplicate assignment where 'kind' is first set to 'quicksort' then overwritten to None.
Falsifying example: test_argsort_defaults_kind_should_not_be_duplicated(
    data=data(...),
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS contents:")
for key, value in ARGSORT_DEFAULTS.items():
    print(f"  {key}: {value!r}")

print(f"\nBug: 'kind' is {ARGSORT_DEFAULTS['kind']!r}, expected 'quicksort'")

# Also show that ARGSORT_DEFAULTS_KIND exists and doesn't have 'kind'
from pandas.compat.numpy.function import ARGSORT_DEFAULTS_KIND

print("\nARGSORT_DEFAULTS_KIND contents (for comparison):")
for key, value in ARGSORT_DEFAULTS_KIND.items():
    print(f"  {key}: {value!r}")

print("\nNote: ARGSORT_DEFAULTS_KIND correctly doesn't have 'kind' key")
print(f"'kind' in ARGSORT_DEFAULTS_KIND: {'kind' in ARGSORT_DEFAULTS_KIND}")
```

<details>

<summary>
ARGSORT_DEFAULTS shows kind=None instead of quicksort
</summary>
```
ARGSORT_DEFAULTS contents:
  axis: -1
  kind: None
  order: None
  stable: None

Bug: 'kind' is None, expected 'quicksort'

ARGSORT_DEFAULTS_KIND contents (for comparison):
  axis: -1
  order: None
  stable: None

Note: ARGSORT_DEFAULTS_KIND correctly doesn't have 'kind' key
'kind' in ARGSORT_DEFAULTS_KIND: False
```
</details>

## Why This Is A Bug

In the source file `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/numpy/function.py` lines 136-141, there is a clear coding error:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138: Correct assignment
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None  # Line 140: DUPLICATE - overwrites line 138!
ARGSORT_DEFAULTS["stable"] = None
```

The `"kind"` key is assigned twice: first to `"quicksort"` (line 138) and then immediately to `None` (line 140). This duplicate assignment overwrites the intended value.

The bug causes the `validate_argsort` CompatValidator to use incorrect default values when validating parameters. Testing confirms this incorrect behavior:
- `validate_argsort((), {'kind': 'quicksort'})` raises an error saying "the 'kind' parameter is not supported"
- `validate_argsort((), {'kind': None})` incorrectly passes validation

This means users who explicitly pass `kind="quicksort"` (which should be the default) will get an unexpected error message claiming the parameter isn't supported, when in reality it's just not matching the incorrectly set default.

## Relevant Context

The pandas compatibility layer validates that numpy-compatible parameters passed to pandas functions match their expected default values. As documented in the module:

> "To ensure that users do not abuse these parameters, validation is performed... Part of that validation includes whether or not the user attempted to pass in non-default values for these extraneous parameters."

The code has two distinct dictionaries:
1. `ARGSORT_DEFAULTS` - Should include all argsort parameters including "kind"
2. `ARGSORT_DEFAULTS_KIND` - A variant without the "kind" parameter (lines 150-156)

Line 140 appears to be a copy-paste error where someone accidentally duplicated the pattern from `ARGSORT_DEFAULTS_KIND` into `ARGSORT_DEFAULTS`. The comment on line 148 confirms these are meant to be "two different signatures of argsort", so the duplicate assignment is clearly unintentional.

NumPy's actual `argsort` signature has `kind=None` as the default (meaning "use the default sorting algorithm"), but pandas intended to validate against `"quicksort"` explicitly as shown by line 138. The similar `SORT_DEFAULTS` dictionary (line 279) also correctly sets `kind="quicksort"` without duplication.

## Proposed Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ def validate_take_with_convert(convert: ndarray | bool | None, args, kwargs) ->
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```