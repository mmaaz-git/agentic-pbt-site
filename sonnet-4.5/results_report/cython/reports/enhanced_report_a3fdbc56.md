# Bug Report: Cython.Build.Dependencies.DistutilsInfo.merge List Aliasing

**Target**: `Cython.Build.Dependencies.DistutilsInfo.merge`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DistutilsInfo.merge` method incorrectly aliases list values from the source object when merging into empty keys, causing unintended shared mutable state between objects.

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

if __name__ == "__main__":
    test_distutils_info_merge_no_aliasing()
```

<details>

<summary>
**Failing input**: `libs=['0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 22, in <module>
    test_distutils_info_merge_no_aliasing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 6, in test_distutils_info_merge_no_aliasing
    def test_distutils_info_merge_no_aliasing(libs):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 14, in test_distutils_info_merge_no_aliasing
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
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 16, in <module>
    assert result.values['libraries'] is not info2.values['libraries'], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Lists should be independent
```
</details>

## Why This Is A Bug

The `merge` method exhibits inconsistent behavior that violates its documented intent. Line 223 contains an explicit comment stating "# Change a *copy* of the list (Trac #845)", indicating that list aliasing was identified as a bug and should be avoided. However, the implementation only creates a copy when the key already exists in `self.values` (lines 224-228). When the key doesn't exist, line 229 directly assigns the reference, creating an alias.

This inconsistency means that modifying a list in the merged object unexpectedly modifies the source object's list. In Cython's build system, where `DistutilsInfo` objects manage build configurations that are frequently merged from multiple sources (base templates, file directives, etc.), this aliasing can cause configuration settings from one extension to leak into others, leading to difficult-to-diagnose build issues.

The bug affects all `transitive_list` type keys: `libraries`, `library_dirs`, `runtime_library_dirs`, `include_dirs`, `extra_compile_args`, `extra_link_args`, and `depends`.

## Relevant Context

The code structure in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py` (lines 221-229) shows the problematic branching logic:

```python
elif type is transitive_list:
    if key in self.values:
        # Change a *copy* of the list (Trac #845)
        all = self.values[key][:]  # Creates a copy
        for v in value:
            if v not in all:
                all.append(v)
        value = all
    self.values[key] = value  # Direct assignment when key doesn't exist
```

The reference to "Trac #845" in the comment suggests this exact issue was previously identified and partially fixed, but the fix was incomplete as it only addressed the case where the key already exists in the destination object.

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -221,6 +221,8 @@ class DistutilsInfo:
             elif type is transitive_list:
                 if key in self.values:
                     # Change a *copy* of the list (Trac #845)
                     all = self.values[key][:]
                     for v in value:
                         if v not in all:
                             all.append(v)
                     value = all
+                else:
+                    value = value[:]  # Create a copy when key doesn't exist
                 self.values[key] = value
```