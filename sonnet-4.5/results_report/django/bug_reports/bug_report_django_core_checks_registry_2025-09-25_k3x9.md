# Bug Report: django.core.checks.registry - Tag Overwrite on Duplicate Registration

**Target**: `django.core.checks.registry.CheckRegistry.register`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When the same check function is registered multiple times with different tags, the second registration silently overwrites the tags from the first registration, making the check no longer callable with the original tags. This violates reasonable user expectations and could cause silent failures in production.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks.registry import CheckRegistry
from django.core.checks import Error


@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10)
)
def test_registry_multiple_registration_different_tags(tag1, tag2):
    """Registering the same check with different tags should preserve all tags"""
    registry = CheckRegistry()

    def my_check(app_configs, **kwargs):
        return [Error("Test error")]

    registry.register(my_check, tag1)
    registry.register(my_check, tag2)

    all_errors = registry.run_checks()
    tag1_errors = registry.run_checks(tags=[tag1])
    tag2_errors = registry.run_checks(tags=[tag2])

    assert len(all_errors) >= 1
    if tag1 != tag2:
        assert len(tag1_errors) >= 1, f"Check registered with tag1={tag1} should be callable with that tag"
        assert len(tag2_errors) >= 1, f"Check registered with tag2={tag2} should be callable with that tag"
```

**Failing input**: `tag1='00'`, `tag2='0'` (or any two different tags)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.core.checks.registry import CheckRegistry
from django.core.checks import Error

registry = CheckRegistry()

def my_check(app_configs, **kwargs):
    return [Error("Test error")]

registry.register(my_check, 'database')
print(f"After first registration, tags: {my_check.tags}")

registry.register(my_check, 'security')
print(f"After second registration, tags: {my_check.tags}")

database_errors = registry.run_checks(tags=['database'])
security_errors = registry.run_checks(tags=['security'])

print(f"Database tag errors: {len(database_errors)}")
print(f"Security tag errors: {len(security_errors)}")

assert len(database_errors) == 0
assert len(security_errors) == 1
```

## Why This Is A Bug

1. **Silent failure**: The first registration appears to succeed, but is silently broken by the second registration
2. **Surprising behavior**: Users would reasonably expect either:
   - An error when attempting to register the same function twice
   - Merging of tags from multiple registrations
   - The second registration to be ignored
3. **Real-world impact**: This could occur when:
   - A check function is imported and registered by multiple modules
   - Someone uses multiple `@register()` decorators thinking it adds both tags
   - Third-party apps register checks that the main app also registers

The root cause is in `registry.py` line 56:
```python
check.tags = tags  # Overwrites previous tags
```

Combined with line 62:
```python
checks.add(check)  # Set prevents duplicate, but tags are already overwritten
```

## Fix

The fix should either prevent duplicate registration or merge tags. Here's a patch that prevents duplicate registration and raises a clear error:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -49,6 +49,11 @@ class CheckRegistry:
         """

         def inner(check):
+            # Check if already registered
+            if check in self.registered_checks or check in self.deployment_checks:
+                raise ValueError(
+                    f"Check function {check.__name__!r} is already registered. "
+                    f"To register with multiple tags, pass them in one call: "
+                    f"register(check, 'tag1', 'tag2')"
+                )
             if not func_accepts_kwargs(check):
                 raise TypeError(
                     "Check functions must accept keyword arguments (**kwargs)."
```

Alternatively, to merge tags from multiple registrations:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -49,7 +49,14 @@ class CheckRegistry:
         """

         def inner(check):
+            checks = (
+                self.deployment_checks
+                if kwargs.get("deploy")
+                else self.registered_checks
+            )
+            # Merge tags if already registered
+            if check in checks:
+                check.tags = check.tags + tags
+            else:
+                check.tags = tags
             if not func_accepts_kwargs(check):
                 raise TypeError(
                     "Check functions must accept keyword arguments (**kwargs)."
                 )
-            check.tags = tags
-            checks = (
-                self.deployment_checks
-                if kwargs.get("deploy")
-                else self.registered_checks
-            )
             checks.add(check)
             return check
```