# Bug Report: django.core.checks.CheckRegistry Double Registration Overwrites Tags

**Target**: `django.core.checks.registry.CheckRegistry.register`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When the same check function is registered multiple times with different tags using `CheckRegistry.register()`, the second registration overwrites the tags from the first registration instead of accumulating them. This means the check function becomes inaccessible via its original tags.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks.registry import CheckRegistry
from django.core.checks import Info


@given(st.text(min_size=1), st.text(min_size=1))
def test_double_registration_preserves_both_tags(tag1, tag2):
    from hypothesis import assume
    assume(tag1 != tag2)

    registry = CheckRegistry()

    def my_check(app_configs=None, **kwargs):
        return [Info("test")]

    registry.register(my_check, tag1)
    registry.register(my_check, tag2)

    available_tags = registry.tags_available()

    assert tag1 in available_tags, f"Tag '{tag1}' should still be available"
    assert tag2 in available_tags, f"Tag '{tag2}' should be available"
```

**Failing input**: `tag1 = "first"`, `tag2 = "second"`

## Reproducing the Bug

```python
from django.core.checks.registry import CheckRegistry
from django.core.checks import Info

registry = CheckRegistry()

def my_check(app_configs=None, **kwargs):
    return [Info("My check")]

registry.register(my_check, "tag1")
print(f"After first registration - tags: {my_check.tags}")

registry.register(my_check, "tag2")
print(f"After second registration - tags: {my_check.tags}")

available_tags = registry.tags_available()
print(f"Available tags: {available_tags}")

checks_tag1 = registry.run_checks(tags=["tag1"])
checks_tag2 = registry.run_checks(tags=["tag2"])

print(f"Checks with tag1: {len(checks_tag1)}")
print(f"Checks with tag2: {len(checks_tag2)}")
```

Output:
```
After first registration - tags: ('tag1',)
After second registration - tags: ('tag2',)
Available tags: {'tag2'}
Checks with tag1: 0
Checks with tag2: 1
```

## Why This Is A Bug

A user might reasonably expect to be able to register the same check function under multiple tags by calling `register()` multiple times. However, the current implementation:

1. Stores checks in a set (line 62: `checks.add(check)`), so adding the same function twice does nothing
2. Overwrites the `tags` attribute (line 56: `check.tags = tags`), losing the previous tags

This means:
- The check is only accessible via the most recently registered tags
- Earlier tags become inaccessible, even though they may still appear to be registered
- This violates the principle of least surprise

## Fix

Instead of overwriting tags, accumulate them when a check is registered multiple times:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -53,7 +53,11 @@ class CheckRegistry:
             raise TypeError(
                 "Check functions must accept keyword arguments (**kwargs)."
             )
-            check.tags = tags
+            if hasattr(check, 'tags'):
+                # Accumulate tags if check is already registered
+                check.tags = tuple(set(check.tags) | set(tags))
+            else:
+                check.tags = tags
             checks = (
                 self.deployment_checks
                 if kwargs.get("deploy")
```

This fix ensures that registering a check multiple times with different tags accumulates all the tags, making the check accessible via any of them.