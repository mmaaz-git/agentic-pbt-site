# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Dead Code Due to Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary contains dead code where line 138 assigns `"kind": "quicksort"` which is immediately overwritten by line 140 assigning `"kind": None`, making the first assignment pointless.

## Property-Based Test

```python
from hypothesis import given, strategies as st

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.just(None))
def test_argsort_defaults_kind_value(dummy):
    """Test that ARGSORT_DEFAULTS['kind'] is None.

    This test passes, but reveals dead code: line 138 sets
    ARGSORT_DEFAULTS['kind'] = 'quicksort' which is immediately
    overwritten by line 140 setting it to None.
    """
    assert ARGSORT_DEFAULTS["kind"] is None

# Run the test
test_argsort_defaults_kind_value()
```

<details>

<summary>
**Failing input**: N/A (static code bug - test passes)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/23
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_argsort_defaults_kind_value PASSED                         [100%]

============================== 1 passed in 0.32s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")

# Show that line 138 is dead code - the value 'quicksort' is never used
# because line 140 immediately overwrites it with None
assert ARGSORT_DEFAULTS["kind"] is None, "Expected kind to be None due to overwrite on line 140"
print("Assertion passed: ARGSORT_DEFAULTS['kind'] is None")
print("\nThis demonstrates that line 138 (setting kind='quicksort') is dead code")
print("because line 140 immediately overwrites it with None")
```

<details>

<summary>
Dead code demonstration - line 138's assignment is overwritten
</summary>
```
ARGSORT_DEFAULTS: {'axis': -1, 'kind': None, 'order': None, 'stable': None}
ARGSORT_DEFAULTS['kind'] = None
Assertion passed: ARGSORT_DEFAULTS['kind'] is None

This demonstrates that line 138 (setting kind='quicksort') is dead code
because line 140 immediately overwrites it with None
```
</details>

## Why This Is A Bug

This violates expected behavior for multiple reasons:

1. **Dead Code Pattern**: Lines 136-141 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/numpy/function.py` show a clear bug:
   ```python
   ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
   ARGSORT_DEFAULTS["axis"] = -1
   ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138 - this assignment...
   ARGSORT_DEFAULTS["order"] = None
   ARGSORT_DEFAULTS["kind"] = None         # Line 140 - ...is immediately overwritten here
   ARGSORT_DEFAULTS["stable"] = None
   ```

2. **NumPy Compatibility Issue**: The module's docstring states it exists "for compatibility with numpy libraries". NumPy's `numpy.argsort` documentation specifies the default for `kind` is `"quicksort"`, not `None`. This breaks the expected compatibility.

3. **Inconsistent with Other Defaults**: Looking at line 279, `SORT_DEFAULTS["kind"] = "quicksort"` correctly sets the default without being overwritten, showing the intended pattern.

4. **Two Validators Pattern Confusion**: The module defines both `ARGSORT_DEFAULTS` (with the buggy `kind=None`) and `ARGSORT_DEFAULTS_KIND` (which omits `kind` entirely). The comment on lines 148-149 explains these are "two different signatures of argsort, this second validation for when the `kind` param is supported", suggesting `ARGSORT_DEFAULTS` should have the proper NumPy-compatible default when `kind` is supported.

## Relevant Context

The pandas compatibility layer validates that users only pass default values for NumPy parameters that pandas doesn't actually use. The module provides default argument dictionaries used throughout the codebase.

In `pandas/core/indexes/range.py`, the code explicitly pops and ignores the `kind` parameter with the comment "e.g. 'mergesort' is irrelevant", then validates with `validate_argsort`. This shows some pandas implementations ignore the `kind` parameter, which explains why this bug hasn't caused runtime errors.

The duplicate assignment appears to be an unintentional copy-paste error or typo during development. The first assignment on line 138 matching NumPy's default suggests that was the original intention.

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

## Proposed Fix

Remove the dead code assignment on line 138 since line 140 overwrites it anyway:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,7 +135,6 @@

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```

Alternatively, for better NumPy compatibility, remove line 140 instead to preserve the NumPy-compatible default:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -137,7 +137,6 @@
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```