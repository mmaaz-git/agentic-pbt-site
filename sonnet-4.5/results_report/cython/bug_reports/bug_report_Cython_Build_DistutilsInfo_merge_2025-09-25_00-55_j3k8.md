# Bug Report: Cython.Build.Dependencies.DistutilsInfo.merge List Aliasing

**Target**: `Cython.Build.Dependencies.DistutilsInfo.merge`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DistutilsInfo.merge` method fails to create copies of `transitive_list` values when the key doesn't exist in `self`. This causes list aliasing where modifications to the merged object's lists also modify the source object's lists, violating the documented intent to "change a *copy* of the list" (as noted in comment on line 224).

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
```

**Failing input**: Any list of libraries, e.g., `libs=['lib1', 'lib2']`

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

Output:
```
Same object: True
info2 libraries: ['lib1', 'lib2', 'lib3']
result libraries: ['lib1', 'lib2', 'lib3']
AssertionError: Lists should be independent
```

## Why This Is A Bug

The comment on line 224 explicitly states: "Change a *copy* of the list (Trac #845)", referencing a previous bug that was fixed. However, this fix only applies when the key already exists in `self.values`.

When the key does NOT exist in `self.values`, the code on line 229 directly assigns the reference:
```python
self.values[key] = value
```

This creates aliasing where both `self.values[key]` and `other.values[key]` point to the same list object. Any modification to one affects the other, which can lead to:

1. **Unintended side effects**: Modifying merged configuration unintentionally modifies the source
2. **Shared mutable state**: Multiple DistutilsInfo objects share the same list, violating expectations
3. **Difficult-to-debug issues**: Changes appear in unexpected places

This is particularly problematic in Cython's build system where DistutilsInfo objects are merged to combine configuration from multiple sources (base templates, file directives, etc.). Aliasing can cause configuration from one extension to leak into others.

## Fix

The fix is to always create a copy of `transitive_list` values, regardless of whether the key exists:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -220,11 +220,12 @@ class DistutilsInfo:
             self.values[key] = value
         elif type is transitive_list:
             if key in self.values:
                 # Change a *copy* of the list (Trac #845)
                 all = self.values[key][:]
                 for v in value:
                     if v not in all:
                         all.append(v)
                 value = all
+            else:
+                value = value[:]  # Create a copy when key doesn't exist
             self.values[key] = value
         elif type is bool_or:
             self.values[key] = self.values.get(key, False) | value
```

Alternatively, a simpler fix that always creates a copy:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -220,6 +220,8 @@ class DistutilsInfo:
             self.values[key] = value
         elif type is transitive_list:
+            # Always work with a copy (Trac #845)
+            value = value[:] if value else []
             if key in self.values:
-                # Change a *copy* of the list (Trac #845)
-                all = self.values[key][:]
+                all = self.values[key]
                 for v in value:
                     if v not in all:
                         all.append(v)
                 value = all
             self.values[key] = value
```