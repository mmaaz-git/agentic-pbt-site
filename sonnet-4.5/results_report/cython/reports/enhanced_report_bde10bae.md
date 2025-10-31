# Bug Report: Cython.Distutils.Extension Parameter Loss with Mixed pyrex/cython Options

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When constructing a Cython Extension with explicit `cython_*` parameters alongside any `pyrex_*` keyword arguments, the explicit `cython_*` parameters are silently discarded and reset to their default values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from Cython.Distutils import Extension


@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
    st.booleans(),
)
@example(include_dirs=['0'], pyrex_gdb=False)  # The specific failing case
def test_extension_explicit_cython_with_pyrex_kwarg(include_dirs, pyrex_gdb):
    ext = Extension(
        "test",
        ["test.pyx"],
        cython_include_dirs=include_dirs,
        pyrex_gdb=pyrex_gdb,
    )

    assert ext.cython_include_dirs == include_dirs
    assert ext.cython_gdb == pyrex_gdb
```

<details>

<summary>
**Failing input**: `include_dirs=['0'], pyrex_gdb=False`
</summary>
```
Testing with specific failing input: include_dirs=['0'], pyrex_gdb=False
Test FAILED: expected cython_include_dirs=['0'], got []
Test FAILED with AssertionError: Extension attributes don't match: include_dirs=[], gdb=False

Running Hypothesis tests to find more failures...
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/55
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29, typeguard-4.3.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_extension_explicit_cython_with_pyrex_kwarg FAILED          [100%]

=================================== FAILURES ===================================
_______________ test_extension_explicit_cython_with_pyrex_kwarg ________________

    @given(
>       st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
                   ^^^
        st.booleans(),
    )

hypo.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

include_dirs = ['0'], pyrex_gdb = False

    @given(
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
        st.booleans(),
    )
    @example(include_dirs=['0'], pyrex_gdb=False)  # The specific failing case
    def test_extension_explicit_cython_with_pyrex_kwarg(include_dirs, pyrex_gdb):
        ext = Extension(
            "test",
            ["test.pyx"],
            cython_include_dirs=include_dirs,
            pyrex_gdb=pyrex_gdb,
        )

>       assert ext.cython_include_dirs == include_dirs
E       AssertionError: assert [] == ['0']
E
E         Right contains one more item: '0'
E
E         Full diff:
E         + []
E         - [
E         -     '0',
E         - ]
E       Falsifying explicit example: test_extension_explicit_cython_with_pyrex_kwarg(
E           include_dirs=['0'],
E           pyrex_gdb=False,
E       )

hypo.py:18: AssertionError
=============================== warnings summary ===============================
../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; _hypothesis_globals
    self._mark_plugins_for_rewrite(hook, disable_autoload)

../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; hypothesis
    self._mark_plugins_for_rewrite(hook, disable_autoload)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ Hypothesis Statistics =============================
=========================== short test summary info ============================
FAILED hypo.py::test_extension_explicit_cython_with_pyrex_kwarg - AssertionEr...
======================== 1 failed, 2 warnings in 0.59s =========================
```
</details>

## Reproducing the Bug

```python
from Cython.Distutils import Extension

# Test case 1: cython_include_dirs with pyrex_gdb
ext1 = Extension(
    "test",
    ["test.pyx"],
    cython_include_dirs=['/my/include/path'],
    pyrex_gdb=True,
)

print("Test 1: cython_include_dirs with pyrex_gdb")
print(f"Expected: ['/my/include/path']")
print(f"Actual:   {ext1.cython_include_dirs}")
print(f"Result:   {'PASS' if ext1.cython_include_dirs == ['/my/include/path'] else 'FAIL'}")
print()

# Test case 2: cython_directives with pyrex_cplus
ext2 = Extension(
    "test2",
    ["test2.pyx"],
    cython_directives={'boundscheck': False},
    pyrex_cplus=True,
)

print("Test 2: cython_directives with pyrex_cplus")
print(f"Expected: {{'boundscheck': False}}")
print(f"Actual:   {ext2.cython_directives}")
print(f"Result:   {'PASS' if ext2.cython_directives == {'boundscheck': False} else 'FAIL'}")
print()

# Test case 3: multiple cython_* parameters with pyrex_* parameter
ext3 = Extension(
    "test3",
    ["test3.pyx"],
    cython_include_dirs=['/path1', '/path2'],
    cython_directives={'language_level': '3'},
    cython_create_listing=True,
    pyrex_gdb=False,
)

print("Test 3: Multiple cython_* parameters with pyrex_gdb")
print(f"Expected include_dirs:    ['/path1', '/path2']")
print(f"Actual include_dirs:      {ext3.cython_include_dirs}")
print(f"Expected directives:      {{'language_level': '3'}}")
print(f"Actual directives:        {ext3.cython_directives}")
print(f"Expected create_listing:  True")
print(f"Actual create_listing:    {ext3.cython_create_listing}")
print(f"Expected gdb:             False")
print(f"Actual gdb:               {ext3.cython_gdb}")
print(f"Result: {'PASS' if ext3.cython_include_dirs == ['/path1', '/path2'] and ext3.cython_directives == {'language_level': '3'} and ext3.cython_create_listing == True and ext3.cython_gdb == False else 'FAIL'}")
```

<details>

<summary>
Output showing all tests fail due to parameter loss
</summary>
```
Test 1: cython_include_dirs with pyrex_gdb
Expected: ['/my/include/path']
Actual:   []
Result:   FAIL

Test 2: cython_directives with pyrex_cplus
Expected: {'boundscheck': False}
Actual:   {}
Result:   FAIL

Test 3: Multiple cython_* parameters with pyrex_gdb
Expected include_dirs:    ['/path1', '/path2']
Actual include_dirs:      []
Expected directives:      {'language_level': '3'}
Actual directives:        {}
Expected create_listing:  True
Actual create_listing:    False
Expected gdb:             False
Actual gdb:               False
Result: FAIL
```
</details>

## Why This Is A Bug

The Extension class provides backward compatibility for the deprecated `pyrex_*` parameter names by translating them to `cython_*` names. However, when any `pyrex_*` parameter is detected, the code makes a recursive call to `Extension.__init__` (lines 47-64 in extension.py) that only passes the distutils base class parameters and `no_c_in_traceback`.

This recursive call completely omits all explicit `cython_*` parameters that the user provided, causing silent data loss. The function returns early after the recursive call (line 64), never reaching the code that would set the cython attributes (lines 83-92).

This violates expected behavior because:
1. **Silent data loss**: User-provided configuration is discarded without any warning or error
2. **Principle of least surprise**: Users expect that mixing old and new parameter syntax would either work correctly or raise an error, not silently ignore their input
3. **Migration hazard**: This bug particularly affects users who are gradually migrating from `pyrex_*` to `cython_*` syntax, a common scenario when updating legacy code

## Relevant Context

The bug exists in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/extension.py`. The `pyrex_*` parameters are legacy from when Cython was called Pyrex. While these are deprecated, they're still supported for backward compatibility.

The issue occurs because the recursive call pattern used for parameter translation doesn't preserve the explicit `cython_*` parameters. When the code detects any `pyrex_*` kwarg, it:
1. Converts all `pyrex_*` kwargs to `cython_*` in the `kw` dict (lines 42-45)
2. Makes a recursive call with only the base parameters and the modified `kw` dict (lines 47-63)
3. Returns immediately, bypassing the attribute assignment code (lines 83-92)

The explicit `cython_*` parameters passed as named arguments are lost because they're not included in the recursive call.

## Proposed Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -44,6 +44,17 @@ class Extension(_Extension.Extension):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
+            # Preserve explicit cython_* parameters in the recursive call
+            if cython_include_dirs is not None:
+                kw.setdefault('cython_include_dirs', cython_include_dirs)
+            if cython_directives is not None:
+                kw.setdefault('cython_directives', cython_directives)
+            if cython_create_listing is not False:
+                kw.setdefault('cython_create_listing', cython_create_listing)
+            if cython_line_directives is not False:
+                kw.setdefault('cython_line_directives', cython_line_directives)
+            if cython_cplus is not False:
+                kw.setdefault('cython_cplus', cython_cplus)
+            if cython_c_in_temp is not False:
+                kw.setdefault('cython_c_in_temp', cython_c_in_temp)
+            if cython_gen_pxi is not False:
+                kw.setdefault('cython_gen_pxi', cython_gen_pxi)
+            if cython_gdb is not False:
+                kw.setdefault('cython_gdb', cython_gdb)
+            if cython_compile_time_env is not None:
+                kw.setdefault('cython_compile_time_env', cython_compile_time_env)
             Extension.__init__(
                 self, name, sources,
                 include_dirs=include_dirs,
```