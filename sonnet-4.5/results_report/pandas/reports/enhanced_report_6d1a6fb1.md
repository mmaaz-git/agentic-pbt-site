# Bug Report: Cython.Build.Dependencies.DistutilsInfo.merge List Aliasing

**Target**: `Cython.Build.Dependencies.DistutilsInfo.merge`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DistutilsInfo.merge` method creates list aliases instead of copies when merging `transitive_list` values that don't exist in the target object, causing unintended shared mutable state between DistutilsInfo instances.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import DistutilsInfo


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5))
def test_distutils_info_merge_no_aliasing(libs):
    info1 = DistutilsInfo()
    info2 = DistutilsInfo()
    info2.values['libraries'] = libs[:]

    result = info1.merge(info2)

    assert result is info1, "merge should modify and return self"
    assert result.values['libraries'] is not info2.values['libraries'], \
        "Merged list should be a copy, not an alias"

    result.values['libraries'].append('new_lib')
    assert 'new_lib' not in info2.values['libraries'], \
        "Modifying merged list should not affect source"

# Run the test
test_distutils_info_merge_no_aliasing()
```

<details>

<summary>
**Failing input**: `libs=['0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 22, in <module>
    test_distutils_info_merge_no_aliasing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 6, in test_distutils_info_merge_no_aliasing
    def test_distutils_info_merge_no_aliasing(libs):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 14, in test_distutils_info_merge_no_aliasing
    assert result.values['libraries'] is not info2.values['libraries'], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Merged list should be a copy, not an alias
Falsifying example: test_distutils_info_merge_no_aliasing(
    libs=['0'],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import DistutilsInfo

info1 = DistutilsInfo()
info2 = DistutilsInfo()
info2.values['libraries'] = ['lib1', 'lib2']

result = info1.merge(info2)

print(f"Same object: {result.values['libraries'] is info2.values['libraries']}")

result.values['libraries'].append('lib3')

print(f"info2 libraries: {info2.values['libraries']}")
print(f"result libraries: {result.values['libraries']}")

assert result.values['libraries'] is not info2.values['libraries'], \
    "Lists should be independent"
```

<details>

<summary>
AssertionError: Lists should be independent
</summary>
```
Same object: True
info2 libraries: ['lib1', 'lib2', 'lib3']
result libraries: ['lib1', 'lib2', 'lib3']
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 16, in <module>
    assert result.values['libraries'] is not info2.values['libraries'], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Lists should be independent
```
</details>

## Why This Is A Bug

The `merge` method violates its documented behavior of working with copies of lists. The comment on line 223 of Dependencies.py explicitly states "Change a *copy* of the list (Trac #845)", indicating that a previous bug (Trac #845) was fixed to ensure lists are copied rather than aliased. However, this fix only applies when the key already exists in `self.values` (lines 222-228).

When the key does NOT exist in `self.values`, the code falls through to line 229 where it directly assigns the reference: `self.values[key] = value`. This creates an alias where both `self.values[key]` and `other.values[key]` point to the same list object.

This behavior contradicts the documented intent and creates several problems:
1. **Unintended side effects**: Modifying a merged DistutilsInfo object's lists also modifies the source object's lists
2. **Shared mutable state**: Multiple DistutilsInfo objects unexpectedly share the same list instances
3. **Configuration leakage**: In Cython's build system where DistutilsInfo objects are merged to combine configuration from multiple sources (base templates, file directives, etc.), this can cause configuration from one extension to leak into others

## Relevant Context

The DistutilsInfo class is used in Cython's build system to manage distutils configuration for building extensions. The `merge` method is designed to combine configuration from multiple sources. The values dictionary can contain different types of settings, including `transitive_list` type values like 'libraries', 'include_dirs', 'library_dirs', etc.

The bug appears to be a regression or incomplete fix from Trac #845, where list copying was implemented only for the case where the key already exists, but not for the initial assignment case.

Code location: `/Cython/Build/Dependencies.py`, lines 221-229
Relevant Cython documentation: https://cython.readthedocs.io/

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -226,6 +226,8 @@ class DistutilsInfo:
                     for v in value:
                         if v not in all:
                             all.append(v)
                     value = all
+                else:
+                    value = value[:]  # Create a copy when key doesn't exist
                 self.values[key] = value
             elif type is bool_or:
                 self.values[key] = self.values.get(key, False) | value
```