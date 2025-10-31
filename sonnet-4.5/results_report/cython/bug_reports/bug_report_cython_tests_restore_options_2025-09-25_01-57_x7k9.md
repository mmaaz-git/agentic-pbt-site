# Bug Report: Cython.Compiler.Tests.Utils.restore_Options Dictionary Mutation During Iteration

**Target**: `Cython.Compiler.Tests.Utils.restore_Options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `restore_Options` function crashes with RuntimeError when deleting newly-added options because it modifies a dictionary while iterating over it.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

def test_backup_restore_options_roundtrip():
    backup = backup_Options()

    Options.new_attribute = "test_value"
    Options.buffer_max_dims = 999

    restore_Options(backup)

    assert not hasattr(Options, 'new_attribute')
    assert check_global_options(backup) == ""
```

**Failing input**: Any call to `restore_Options` after adding new attributes to `Options`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, 'lib/python3.13/site-packages')

from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

backup = backup_Options()
Options.new_attribute = "test_value"
restore_Options(backup)
```

Output: `RuntimeError: dictionary changed size during iteration`

## Why This Is A Bug

At line 22-24 of Utils.py, the code iterates over dictionary keys and deletes from the same dictionary:

```python
for name in vars(Options).keys():
    if name not in backup:
        delattr(Options, name)
```

This is a classic Python error - modifying a dictionary during iteration causes RuntimeError. This bug prevents proper cleanup of test state when new options are added.

## Fix

```diff
--- a/Cython/Compiler/Tests/Utils.py
+++ b/Cython/Compiler/Tests/Utils.py
@@ -19,7 +19,7 @@ def restore_Options(backup):
         if getattr(Options, name, no_value) != orig_value:
             setattr(Options, name, orig_value)
     # strip Options from new keys that might have been added:
-    for name in vars(Options).keys():
+    for name in list(vars(Options).keys()):
         if name not in backup:
             delattr(Options, name)
```