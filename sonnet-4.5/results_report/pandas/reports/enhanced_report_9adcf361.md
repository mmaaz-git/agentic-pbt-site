# Bug Report: Cython.Plex.Actions.Call.__repr__ Crashes on Callable Objects Without __name__ Attribute

**Target**: `Cython.Plex.Actions.Call.__repr__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Call.__repr__` method crashes with an AttributeError when used with callable objects that lack a `__name__` attribute, such as callable class instances or functools.partial objects.

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

<details>

<summary>
**Failing input**: `value=0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/26
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_call_repr_with_callable_object FAILED

=================================== FAILURES ===================================
_____________________ test_call_repr_with_callable_object ______________________

    @given(st.integers())
>   def test_call_repr_with_callable_object(value):
                   ^^^

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:11: in test_call_repr_with_callable_object
    repr_str = repr(action)
               ^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   AttributeError: 'MyCallable' object has no attribute '__name__'. Did you mean: '__ne__'?
E   Falsifying example: test_call_repr_with_callable_object(
E       value=0,
E   )

Cython/Plex/Actions.py:46: AttributeError
=========================== short test summary info ============================
FAILED hypo.py::test_call_repr_with_callable_object - AttributeError: 'MyCall...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 0.13s ===============================
```
</details>

## Reproducing the Bug

```python
from Cython.Plex.Actions import Call
import functools

# Test Case 1: Callable object (class with __call__ method)
class CallableObject:
    def __call__(self, scanner, text):
        return 'result'

print("Test 1: Callable object")
try:
    action1 = Call(CallableObject())
    print(repr(action1))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 2: functools.partial
def base_func(scanner, text, extra):
    return extra

print("\nTest 2: functools.partial")
try:
    action2 = Call(functools.partial(base_func, extra=10))
    print(repr(action2))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 3: Lambda function (should work)
print("\nTest 3: Lambda function")
try:
    action3 = Call(lambda scanner, text: 'lambda_result')
    print(repr(action3))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 4: Regular function (should work)
def regular_func(scanner, text):
    return 'regular_result'

print("\nTest 4: Regular function")
try:
    action4 = Call(regular_func)
    print(repr(action4))
except AttributeError as e:
    print(f"AttributeError: {e}")
```

<details>

<summary>
AttributeError on callable objects and functools.partial
</summary>
```
Test 1: Callable object
AttributeError: 'CallableObject' object has no attribute '__name__'

Test 2: functools.partial
AttributeError: 'functools.partial' object has no attribute '__name__'

Test 3: Lambda function
Call(<lambda>)

Test 4: Regular function
Call(regular_func)
```
</details>

## Why This Is A Bug

The `Call.__repr__` method in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Actions.py` at line 46 unconditionally accesses `self.function.__name__`:

```python
def __repr__(self):
    return "Call(%s)" % self.function.__name__
```

This violates the implicit contract of the `Call` class, which accepts ANY callable object. The bug manifests because:

1. **The Call class is designed to accept any callable**: The `__init__` method at line 39 accepts any `function` parameter without validation, and the `perform` method at line 42-43 only requires that `self.function` be callable with two arguments.

2. **Lexicons.py explicitly creates Call actions for any callable**: In `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py` at lines 153-158, the code specifically checks for the presence of a `__call__` attribute and creates a `Call` action for ANY object that has it:
   ```python
   try:
       action_spec.__call__
   except AttributeError:
       action = Actions.Return(action_spec)
   else:
       action = Actions.Call(action_spec)
   ```

3. **Not all callables have a `__name__` attribute**: While regular functions and lambdas have `__name__`, many valid Python callables do not:
   - Class instances with `__call__` method
   - `functools.partial` objects
   - Built-in callable types
   - Custom callable descriptors

4. **The documentation doesn't restrict callables**: The class docstring simply states it "causes a function to be called" but doesn't mandate that only functions with `__name__` are supported. The Lexicon documentation at lines 66-74 also refers to "a function" but doesn't specify this restriction.

5. **Other Action classes handle __repr__ more robustly**: The `Return` class (line 31) uses `repr()` on its value, and the `Method` class (lines 64-68) carefully handles its attributes, showing that defensive coding in `__repr__` is expected.

## Relevant Context

- **Cython.Plex** is a lexical analysis module used by the Cython compiler for tokenization
- The `Call` action is created automatically by the lexer when any callable is provided as a token action
- This bug would affect any Cython user who uses callable objects or partial functions as lexer actions
- The crash only occurs when `repr()` is called (e.g., during debugging or logging), not during normal lexer operation
- The fix is straightforward and maintains backward compatibility

## Proposed Fix

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