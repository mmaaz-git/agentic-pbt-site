# Bug Report: Cython.Plex.Actions.Call.__repr__ Crashes on Callable Objects

**Target**: `Cython.Plex.Actions.Call.__repr__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Call.__repr__` method assumes all callables have a `__name__` attribute, causing crashes when used with callable objects, functools.partial, or other callables lacking this attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Plex.Actions import Call

@given(st.integers())
def test_call_repr_with_callable_object(value):
    class MyCallable:
        def __call__(self, scanner, text):
            return value

    action = Call(MyCallable())
    repr_str = repr(action)
    assert 'Call' in repr_str
```

**Failing input**: Any value (e.g., `value=0`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, 'lib/python3.13/site-packages')

from Cython.Plex.Actions import Call
import functools

class CallableObject:
    def __call__(self, scanner, text):
        return 'result'

action1 = Call(CallableObject())
print(repr(action1))

def base_func(scanner, text, extra):
    return extra

action2 = Call(functools.partial(base_func, extra=10))
print(repr(action2))
```

Output:
```
AttributeError: 'CallableObject' object has no attribute '__name__'
```

## Why This Is A Bug

The `__repr__` method unconditionally accesses `self.function.__name__` at line 46 of Actions.py:

```python
def __repr__(self):
    return "Call(%s)" % self.function.__name__
```

However, not all callables have a `__name__` attribute:
- Callable objects (instances with `__call__` method)
- functools.partial objects
- Other callable types

Since `Call.__init__` accepts any callable (line 39), and Lexicon creates Call actions for any callable (Lexicons.py:158), these are valid inputs that should be supported.

## Fix

Use `getattr` with a fallback to handle callables without `__name__`:

```diff
--- a/Cython/Plex/Actions.py
+++ b/Cython/Plex/Actions.py
@@ -43,7 +43,7 @@ class Call(Action):
         return self.function(token_stream, text)

     def __repr__(self):
-        return "Call(%s)" % self.function.__name__
+        return "Call(%s)" % getattr(self.function, '__name__', repr(self.function))
```