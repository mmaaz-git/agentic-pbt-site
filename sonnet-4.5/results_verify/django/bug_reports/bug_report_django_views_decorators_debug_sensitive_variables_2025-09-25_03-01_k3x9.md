# Bug Report: django.views.decorators.debug.sensitive_variables Inconsistent Wrapping Behavior

**Target**: `django.views.decorators.debug.sensitive_variables`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sensitive_variables` decorator exhibits inconsistent behavior between synchronous and asynchronous functions. For sync functions, it creates a wrapper function and sets an attribute on it. For async functions, it returns the original function unchanged and stores metadata in a global dictionary. This violates the decorator contract principle that decorators should behave consistently regardless of whether they decorate sync or async functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.views.decorators.debug import sensitive_variables


@given(var_name=st.text(min_size=1, max_size=20))
def test_sensitive_variables_wrapping_consistency(var_name):
    def sync_func(request):
        return "sync"

    async def async_func(request):
        return "async"

    decorated_sync = sensitive_variables(var_name)(sync_func)
    decorated_async = sensitive_variables(var_name)(async_func)

    sync_is_wrapped = decorated_sync is not sync_func
    async_is_wrapped = decorated_async is not async_func

    assert sync_is_wrapped == async_is_wrapped, \
        "Decorator should wrap both sync and async functions consistently"
```

**Failing input**: Any variable name (e.g., `'password'`)

## Reproducing the Bug

```python
from django.views.decorators.debug import sensitive_variables


def my_sync_view(request):
    password = "secret"
    return "response"


async def my_async_view(request):
    password = "secret"
    return "response"


decorated_sync = sensitive_variables('password')(my_sync_view)
decorated_async = sensitive_variables('password')(my_async_view)

print(f"Sync wrapped: {decorated_sync is not my_sync_view}")
print(f"Async wrapped: {decorated_async is not my_async_view}")

print(f"Sync has attribute: {hasattr(decorated_sync, 'sensitive_variables')}")
print(f"Async has attribute: {hasattr(decorated_async, 'sensitive_variables')}")
```

Expected output (consistent behavior):
```
Sync wrapped: True
Async wrapped: True
Sync has attribute: True
Async has attribute: True
```

Actual output (inconsistent behavior):
```
Sync wrapped: True
Async wrapped: False
Sync has attribute: True
Async has attribute: False
```

## Why This Is A Bug

1. **Inconsistent decorator contract**: Python decorators should behave consistently. A decorator that wraps sync functions should also wrap async functions.

2. **API inconsistency**: Code that checks `hasattr(func, 'sensitive_variables')` will work for sync functions but fail for async functions, even though both are decorated identically from the user's perspective.

3. **Violates user expectations**: Users applying `@sensitive_variables('password')` to both sync and async views would reasonably expect the same wrapping behavior.

4. **Testing and introspection issues**: Tools that introspect decorated functions will see different behaviors for sync vs async, making it harder to write generic code that works with both.

## Fix

The async code path should create a wrapper function like the sync path does, while still storing metadata in the global dictionary if needed for async-specific processing. Here's a suggested fix:

```diff
--- a/django/views/decorators/debug.py
+++ b/django/views/decorators/debug.py
@@ -40,6 +40,7 @@ def sensitive_variables(*variables):
     def decorator(func):
         if iscoroutinefunction(func):
-            sensitive_variables_wrapper = func
+            # Store metadata in global dict for async processing

             wrapped_func = func
             while getattr(wrapped_func, "__wrapped__", None) is not None:
@@ -63,7 +64,16 @@ def sensitive_variables(*variables):
                 coroutine_functions_to_sensitive_variables[key] = variables
             else:
                 coroutine_functions_to_sensitive_variables[key] = "__ALL__"
+
+            # Create wrapper to maintain consistent decorator behavior
+            @wraps(func)
+            async def sensitive_variables_wrapper(*func_args, **func_kwargs):
+                return await func(*func_args, **func_kwargs)
+
+            # Optionally set attribute for consistency with sync functions
+            if variables:
+                sensitive_variables_wrapper.sensitive_variables = variables
+            else:
+                sensitive_variables_wrapper.sensitive_variables = "__ALL__"

         else:
```

This maintains the global dictionary storage for async functions (which may be needed for async-specific error handling) while also creating a wrapper function for consistency with the sync path.