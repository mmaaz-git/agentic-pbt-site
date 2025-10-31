# Bug Report: Cython.Compiler.Tests.Utils restore_Options Dictionary Iteration Error

**Target**: `Cython.Compiler.Tests.Utils.restore_Options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `restore_Options` function crashes with `RuntimeError: dictionary changed size during iteration` when removing attributes that were added to the Options module after a backup was created, violating Python's fundamental rule against modifying a dictionary during iteration.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options


@given(st.text(min_size=1), st.integers())
def test_backup_restore_round_trip_with_additions(key, value):
    original_backup = backup_Options()
    setattr(Options, key, value)
    restore_Options(original_backup)
    assert not hasattr(Options, key) or getattr(Options, key) == original_backup.get(key)

if __name__ == "__main__":
    test_backup_restore_round_trip_with_additions()
```

<details>

<summary>
**Failing input**: `key='0', value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 14, in <module>
    test_backup_restore_round_trip_with_additions()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 7, in test_backup_restore_round_trip_with_additions
    def test_backup_restore_round_trip_with_additions(key, value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 10, in test_backup_restore_round_trip_with_additions
    restore_Options(original_backup)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/Tests/Utils.py", line 22, in restore_Options
    for name in vars(Options).keys():
                ~~~~~~~~~~~~~~~~~~^^
RuntimeError: dictionary changed size during iteration
Falsifying example: test_backup_restore_round_trip_with_additions(
    # The test always failed when commented parts were varied together.
    key='0',  # or any other generated value
    value=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

original_backup = backup_Options()
setattr(Options, 'new_test_key', 'new_test_value')

restore_Options(original_backup)
```

<details>

<summary>
RuntimeError when restoring Options after adding new attributes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/repo.py", line 7, in <module>
    restore_Options(original_backup)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/Tests/Utils.py", line 22, in restore_Options
    for name in vars(Options).keys():
                ~~~~~~~~~~~~~~~~~~^^
RuntimeError: dictionary changed size during iteration
```
</details>

## Why This Is A Bug

This violates Python's fundamental constraint that dictionaries cannot be modified while being iterated over. The function contains an explicit comment on line 21 stating "strip Options from new keys that might have been added", indicating this is the exact intended behavior. However, the implementation on lines 22-24 iterates directly over `vars(Options).keys()` while calling `delattr(Options, name)` inside the loop, which modifies the underlying dictionary.

The function is specifically designed to handle test isolation by restoring the Options module to its original state after tests may have added new attributes. This crash occurs in the exact scenario the function was created to handle, making it completely unusable for its primary purpose. While this is test infrastructure code rather than core Cython functionality, broken test utilities can lead to test pollution between test cases, making tests flaky and hard to debug.

## Relevant Context

This bug was discovered in Cython version 3.1.4. The `restore_Options` function is part of the test utilities in `/Cython/Compiler/Tests/Utils.py` and is used alongside `backup_Options()` to provide test isolation by saving and restoring the state of the Options module.

The Python documentation explicitly states that "RuntimeError will be raised if the dictionary changes size during iteration", which is exactly what happens here. This is documented behavior in Python's dictionary implementation: https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects

The standard Python idiom to avoid this issue is to create a list copy of the keys before iteration, which is a well-established pattern used throughout the Python ecosystem.

## Proposed Fix

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