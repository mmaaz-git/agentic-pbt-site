# Bug Report: Cython.Build.Dependencies.extended_iglob Deprecation Warning on Python 3.13+

**Target**: `Cython.Build.Dependencies.extended_iglob`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `extended_iglob` function in Cython uses deprecated positional argument syntax for `re.split()` that generates `DeprecationWarning` in Python 3.13+. This will cause the function to fail completely in future Python versions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import extended_iglob
import warnings


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/*', min_size=5, max_size=50))
@settings(max_examples=500)
def test_extended_iglob_no_warnings(pattern):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(extended_iglob(pattern))
        assert len(w) == 0, f"extended_iglob should not generate warnings: {[str(x.message) for x in w]}"

if __name__ == "__main__":
    test_extended_iglob_no_warnings()
```

<details>

<summary>
**Failing input**: `'a**/a'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/7
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_extended_iglob_no_warnings FAILED                          [100%]

=================================== FAILURES ===================================
_______________________ test_extended_iglob_no_warnings ________________________

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/*', min_size=5, max_size=50))
>   @settings(max_examples=500)
                   ^^^

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

pattern = 'a**/a'

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/*', min_size=5, max_size=50))
    @settings(max_examples=500)
    def test_extended_iglob_no_warnings(pattern):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(extended_iglob(pattern))
>           assert len(w) == 0, f"extended_iglob should not generate warnings: {[str(x.message) for x in w]}"
E           AssertionError: extended_iglob should not generate warnings: ["'maxsplit' is passed as positional argument"]
E           assert 1 == 0
E            +  where 1 = len([<warnings.WarningMessage object at 0x75d8837d2990>])
E           Falsifying example: test_extended_iglob_no_warnings(
E               pattern='a**/a',
E           )
E           Explanation:
E               These lines were always and only run by failing examples:
E                   /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:54

hypo.py:12: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_extended_iglob_no_warnings - AssertionError: extended_ig...
============================== 1 failed in 0.40s ===============================
```
</details>

## Reproducing the Bug

```python
import warnings
from Cython.Build.Dependencies import extended_iglob

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    list(extended_iglob('**/*.py'))
    for warning in w:
        print(f'{warning.category.__name__}: {warning.message}')
```

<details>

<summary>
Output: DeprecationWarning raised
</summary>
```
DeprecationWarning: 'maxsplit' is passed as positional argument
DeprecationWarning: 'maxsplit' is passed as positional argument
```
</details>

## Why This Is A Bug

Python 3.13 officially deprecated passing `maxsplit` as a positional argument to `re.split()`. According to the [Python documentation](https://docs.python.org/3/library/re.html#re.split):

> "Deprecated since version 3.13: Passing maxsplit and flags as positional arguments is deprecated. In future Python versions they will be keyword-only parameters."

The `extended_iglob` function at line 55 of `Cython/Build/Dependencies.py` uses:
```python
first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, 1)
```

This passes `1` as a positional argument for `maxsplit`, which triggers the deprecation warning on Python 3.13+. While this currently only produces warnings, it will cause the function to raise a `TypeError` in future Python versions when the deprecation becomes an error.

This violates the principle that library code should not generate deprecation warnings from their internals during normal operation. Users of Cython should be able to use the library without seeing warnings about Cython's internal implementation choices.

## Relevant Context

- The `extended_iglob` function is used internally by Cython's build system to process glob patterns when searching for source files
- The function extends Python's standard `iglob` to support recursive directory matching with `**/` patterns
- This issue affects all users of Cython on Python 3.13+ who use recursive glob patterns in their build configurations
- The deprecation warning appears twice in the test output because the function recursively calls itself on line 61 and 65

Code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:55`

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -52,7 +52,7 @@ def extended_iglob(pattern):
     # because '/' is generally common for relative paths.
     if '**/' in pattern or os.sep == '\\' and '**\\' in pattern:
         seen = set()
-        first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, 1)
+        first, rest = re.split(r'\*\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, maxsplit=1)
         if first:
             first = iglob(first + os.sep)
         else:
```