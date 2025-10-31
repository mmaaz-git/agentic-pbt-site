# Bug Report: pandas.core.computation.scope.Scope.swapkey Fails to Remove Old Key

**Target**: `pandas.core.computation.scope.Scope.swapkey`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Scope.swapkey()` method is documented to "Replace a variable name" but it only adds the new key without removing the old key, violating its contract and causing both keys to coexist in the scope.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.computation.scope import Scope


@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10),
    st.integers(),
    st.integers()
)
def test_swapkey_removes_old_key(old_key, new_key, old_value, new_value):
    """
    Property: After swapkey(old_key, new_key, new_value):
    - new_key should exist with new_value
    - old_key should NOT exist (it should be removed)
    """
    assume(old_key != new_key)

    scope = Scope(level=0, global_dict={old_key: old_value})

    assert old_key in scope.scope
    assert scope.scope[old_key] == old_value

    scope.swapkey(old_key, new_key, new_value)

    assert new_key in scope.scope
    assert scope.scope[new_key] == new_value

    assert old_key not in scope.scope
```

**Failing input**: `old_key='0', new_key='5', old_value=0, new_value=0`

## Reproducing the Bug

```python
from pandas.core.computation.scope import Scope

scope = Scope(level=0, global_dict={'old_key': 'old_value'})

print(f"Before: 'old_key' in scope.scope = {'old_key' in scope.scope}")
print(f"Before: 'new_key' in scope.scope = {'new_key' in scope.scope}")

scope.swapkey('old_key', 'new_key', 'new_value')

print(f"After: 'old_key' in scope.scope = {'old_key' in scope.scope}")
print(f"After: 'new_key' in scope.scope = {'new_key' in scope.scope}")
print(f"After: scope.scope['new_key'] = {scope.scope['new_key']}")

assert 'old_key' not in scope.scope
```

Output:
```
Before: 'old_key' in scope.scope = True
Before: 'new_key' in scope.scope = False
After: 'old_key' in scope.scope = True
After: 'new_key' in scope.scope = True
After: scope.scope['new_key'] = new_value
AssertionError
```

## Why This Is A Bug

The method is named `swapkey` and its docstring states "Replace a variable name, with a potentially new value." The word "Replace" clearly implies that the old key should be removed after adding the new key. However, the implementation only adds `new_key` without removing `old_key`, causing both to exist in the scope.

This violates the method's contract and causes namespace pollution where old variable names persist when they should have been replaced.

## Fix

```diff
--- a/pandas/core/computation/scope.py
+++ b/pandas/core/computation/scope.py
@@ -266,6 +266,8 @@ class Scope:
         for mapping in maps:
             if old_key in mapping:
                 mapping[new_key] = new_value
+                if old_key != new_key:
+                    del mapping[old_key]
                 return
```