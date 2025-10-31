# Bug Report: numpy.typing NBitBase Deprecation Warning Never Fires

**Target**: `numpy.typing.__getattr__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` in `numpy.typing` never fires during normal usage because `NBitBase` is directly imported into the module namespace, preventing `__getattr__` from being called when users access `npt.NBitBase`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for NBitBase deprecation warning."""

import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attributes_trigger_warnings(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)
        assert len(w) >= 1, f"Expected deprecation warning for {attr_name} but got none"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

if __name__ == "__main__":
    test_deprecated_attributes_trigger_warnings()
```

<details>

<summary>
**Failing input**: `"NBitBase"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 17, in <module>
    test_deprecated_attributes_trigger_warnings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 9, in test_deprecated_attributes_trigger_warnings
    def test_deprecated_attributes_trigger_warnings(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 13, in test_deprecated_attributes_trigger_warnings
    assert len(w) >= 1, f"Expected deprecation warning for {attr_name} but got none"
           ^^^^^^^^^^^
AssertionError: Expected deprecation warning for NBitBase but got none
Falsifying example: test_deprecated_attributes_trigger_warnings(
    attr_name='NBitBase',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstration of NBitBase deprecation warning not firing."""

import warnings
import numpy.typing as npt

print("Test 1: Accessing npt.NBitBase directly (normal usage)")
print("-" * 60)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Number of warnings captured: {len(w)}")
    if len(w) > 0:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("NO WARNINGS ISSUED - This is the bug!")

print("\n" + "=" * 60 + "\n")

print("Test 2: Accessing NBitBase via __getattr__ (forced)")
print("-" * 60)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__('NBitBase')
    print(f"Number of warnings captured: {len(w)}")
    if len(w) > 0:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("No warnings issued")

print("\n" + "=" * 60 + "\n")

print("Verification: Why does this happen?")
print("-" * 60)
print(f"NBitBase in npt.__all__: {'NBitBase' in npt.__all__}")
print(f"NBitBase in module globals: {'NBitBase' in dir(npt)}")
print(f"NBitBase is directly accessible: {hasattr(npt, 'NBitBase')}")
print("\nExplanation: Since NBitBase is directly imported into the module")
print("namespace, Python finds it immediately without calling __getattr__.")
print("The deprecation warning code in __getattr__ is therefore unreachable.")
```

<details>

<summary>
NBitBase deprecation warning fails to trigger on normal access but works when forced through __getattr__
</summary>
```
Test 1: Accessing npt.NBitBase directly (normal usage)
------------------------------------------------------------
Number of warnings captured: 0
NO WARNINGS ISSUED - This is the bug!

============================================================

Test 2: Accessing NBitBase via __getattr__ (forced)
------------------------------------------------------------
Number of warnings captured: 1
Warning: DeprecationWarning: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)

============================================================

Verification: Why does this happen?
------------------------------------------------------------
NBitBase in npt.__all__: True
NBitBase in module globals: True
NBitBase is directly accessible: True

Explanation: Since NBitBase is directly imported into the module
namespace, Python finds it immediately without calling __getattr__.
The deprecation warning code in __getattr__ is therefore unreachable.
```
</details>

## Why This Is A Bug

This violates the documented behavior and developer intent for NBitBase deprecation. The code at `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/typing/__init__.py:173-184` contains explicit logic to issue a deprecation warning when NBitBase is accessed, with a comment stating "Deprecated in NumPy 2.3, 2025-05-01". However, this warning mechanism fails because:

1. **Python's attribute resolution order**: In Python, `__getattr__` is only called when an attribute is NOT found through normal attribute lookup. It's a fallback mechanism, not an interception mechanism.

2. **NBitBase is directly imported**: Line 160 imports NBitBase directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`

3. **NBitBase is in __all__**: Line 162 includes NBitBase in the public API: `__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]`

4. **Impact on users**: Without deprecation warnings, users have no programmatic indication that NBitBase is deprecated. Their code will continue to work silently until NBitBase is removed in a future version, at which point their code will suddenly break with an ImportError or AttributeError.

This contradicts NumPy's deprecation policy and semantic versioning practices, which require adequate warning before removing public API elements.

## Relevant Context

The NumPy documentation (referenced in the module docstring at lines 94-112) explicitly discusses NBitBase as part of the "Number precision" section, indicating it's a documented public API feature. The deprecation warning message itself recommends using "@typing.overload or a TypeVar with a scalar-type as upper bound, instead", showing clear migration guidance that users are meant to receive but currently don't.

The bug affects all users who import and use `numpy.typing.NBitBase` in their type annotations. This is particularly problematic for large codebases that rely on deprecation warnings for migration planning.

Documentation reference: https://numpy.org/doc/stable/reference/typing.html

## Proposed Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,9 +157,11 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+# Import NBitBase privately so __getattr__ can handle public access
+from numpy._typing import NBitBase as _NBitBase_imported

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -181,7 +183,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase_imported

     if name in __DIR_SET:
         return globals()[name]
```