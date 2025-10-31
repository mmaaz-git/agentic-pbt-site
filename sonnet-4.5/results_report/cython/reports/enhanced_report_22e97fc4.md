# Bug Report: Cython.Compiler.Tests.Utils.restore_Options Dictionary Mutation During Iteration

**Target**: `Cython.Compiler.Tests.Utils.restore_Options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `restore_Options` function crashes with a RuntimeError when it attempts to remove newly-added attributes from the Options object, because it modifies a dictionary while iterating over its keys.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, Phase
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from Cython.Compiler import Options

@given(attr_name=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier() and not x.startswith('_')),
       attr_value=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans(), st.none()))
@settings(phases=(Phase.generate, Phase.target, Phase.shrink), max_examples=10)
def test_backup_restore_options_roundtrip(attr_name, attr_value):
    """Test that backup_Options and restore_Options properly handle new attributes."""
    # Create a backup of current state
    backup = backup_Options()

    # Add a new attribute
    setattr(Options, attr_name, attr_value)

    # Verify the attribute exists
    assert hasattr(Options, attr_name)
    assert getattr(Options, attr_name) == attr_value

    # Restore to original state
    restore_Options(backup)

    # Verify the new attribute was removed
    assert not hasattr(Options, attr_name), f"New attribute '{attr_name}' should have been removed"

    # Verify all original options are restored
    assert check_global_options(backup) == "", "Original options should be restored"

if __name__ == "__main__":
    test_backup_restore_options_roundtrip()
```

<details>

<summary>
**Failing input**: `attr_name='A', attr_value=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 33, in <module>
    test_backup_restore_options_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 9, in test_backup_restore_options_roundtrip
    attr_value=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans(), st.none()))
            ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 24, in test_backup_restore_options_roundtrip
    restore_Options(backup)
    ~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Tests/Utils.py", line 22, in restore_Options
    for name in vars(Options).keys():
                ~~~~~~~~~~~~~~~~~~^^
RuntimeError: dictionary changed size during iteration
Falsifying example: test_backup_restore_options_roundtrip(
    attr_name='A',
    attr_value=None,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

# Create a backup of the current Options state
backup = backup_Options()

# Add a new attribute to Options
Options.new_attribute = "test_value"

# Try to restore the Options to its original state
# This should remove the new_attribute we just added
restore_Options(backup)
```

<details>

<summary>
RuntimeError: dictionary changed size during iteration
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/repo.py", line 15, in <module>
    restore_Options(backup)
    ~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Tests/Utils.py", line 22, in restore_Options
    for name in vars(Options).keys():
                ~~~~~~~~~~~~~~~~~~^^
RuntimeError: dictionary changed size during iteration
```
</details>

## Why This Is A Bug

This is a clear violation of Python's fundamental rule against modifying a dictionary while iterating over it. The code at lines 22-24 of Utils.py attempts to delete attributes from the Options object while iterating over those same attributes:

```python
for name in vars(Options).keys():
    if name not in backup:
        delattr(Options, name)
```

The comment immediately above this code (line 21) explicitly states the intention: "strip Options from new keys that might have been added". This shows the functionality is intentional and expected to work, but the implementation contains a well-known Python anti-pattern that causes RuntimeError in Python 3. The function cannot fulfill its documented purpose of removing newly-added attributes during test cleanup.

## Relevant Context

The `restore_Options` function is part of Cython's test utilities located in `Cython/Compiler/Tests/Utils.py`. These utilities are designed to backup and restore the global Options state during testing to ensure proper test isolation. The companion function `backup_Options()` creates a dictionary snapshot of all Options attributes, and `restore_Options(backup)` is meant to:

1. Restore all original attribute values from the backup
2. Remove any new attributes that were added after the backup was created

The second part fails due to the dictionary mutation error. This bug could lead to test pollution where Options attributes from one test leak into subsequent tests, potentially causing false test failures or hiding real issues.

The error occurs consistently in Python 3 whenever `delattr` is called within the iteration loop, as it modifies the underlying dictionary returned by `vars(Options)`.

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