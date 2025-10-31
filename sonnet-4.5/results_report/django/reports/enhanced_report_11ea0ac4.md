# Bug Report: django.core.checks.registry Silent Tag Overwrite on Duplicate Check Registration

**Target**: `django.core.checks.registry.CheckRegistry.register`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When the same check function is registered multiple times with different tags in Django's check registry, the second registration silently overwrites the tags from the first registration, making the check unreachable via the original tags.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

# Create minimal Django settings
with open('test_settings.py', 'w') as f:
    f.write('''
SECRET_KEY = 'test-secret-key'
DEBUG = True
INSTALLED_APPS = []
''')

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

if __name__ == "__main__":
    test_registry_multiple_registration_different_tags()
```

<details>

<summary>
**Failing input**: `tag1='0'`, `tag2='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 43, in <module>
    test_registry_multiple_registration_different_tags()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 20, in test_registry_multiple_registration_different_tags
    st.text(min_size=1, max_size=10),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 39, in test_registry_multiple_registration_different_tags
    assert len(tag1_errors) >= 1, f"Check registered with tag1={tag1} should be callable with that tag"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Check registered with tag1=0 should be callable with that tag
Falsifying example: test_registry_multiple_registration_different_tags(
    tag1='0',  # or any other generated value
    tag2='00',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/22/hypo.py:39
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

# Create minimal Django settings
with open('test_settings.py', 'w') as f:
    f.write('''
SECRET_KEY = 'test-secret-key'
DEBUG = True
INSTALLED_APPS = []
''')

from django.core.checks.registry import CheckRegistry
from django.core.checks import Error

registry = CheckRegistry()

def my_check(app_configs, **kwargs):
    return [Error("Test error")]

# Register the same check function with 'database' tag
registry.register(my_check, 'database')
print(f"After first registration with 'database' tag:")
print(f"  check.tags = {my_check.tags}")
print(f"  Tags available in registry: {registry.tags_available()}")

# Register the same check function again with 'security' tag
registry.register(my_check, 'security')
print(f"\nAfter second registration with 'security' tag:")
print(f"  check.tags = {my_check.tags}")
print(f"  Tags available in registry: {registry.tags_available()}")

# Try to run checks with the 'database' tag
database_errors = registry.run_checks(tags=['database'])
print(f"\nRunning checks with 'database' tag:")
print(f"  Number of errors returned: {len(database_errors)}")

# Try to run checks with the 'security' tag
security_errors = registry.run_checks(tags=['security'])
print(f"\nRunning checks with 'security' tag:")
print(f"  Number of errors returned: {len(security_errors)}")

# Run all checks without filtering by tags
all_errors = registry.run_checks()
print(f"\nRunning all checks (no tag filter):")
print(f"  Number of errors returned: {len(all_errors)}")

print("\n" + "="*50)
print("EXPECTED BEHAVIOR:")
print("  - Both 'database' and 'security' tags should work")
print("  - Each tag should return 1 error when used")
print("\nACTUAL BEHAVIOR:")
print(f"  - 'database' tag returns {len(database_errors)} errors (expected: 1)")
print(f"  - 'security' tag returns {len(security_errors)} errors (expected: 1)")
print(f"  - The 'database' tag was silently overwritten!")
```

<details>

<summary>
Output showing tag overwrite behavior
</summary>
```
After first registration with 'database' tag:
  check.tags = ('database',)
  Tags available in registry: {'database'}

After second registration with 'security' tag:
  check.tags = ('security',)
  Tags available in registry: {'security'}

Running checks with 'database' tag:
  Number of errors returned: 0

Running checks with 'security' tag:
  Number of errors returned: 1

Running all checks (no tag filter):
  Number of errors returned: 1

==================================================
EXPECTED BEHAVIOR:
  - Both 'database' and 'security' tags should work
  - Each tag should return 1 error when used

ACTUAL BEHAVIOR:
  - 'database' tag returns 0 errors (expected: 1)
  - 'security' tag returns 1 errors (expected: 1)
  - The 'database' tag was silently overwritten!
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Silent Data Loss**: The first registration appears successful but is silently broken when the second registration occurs. The `check.tags` attribute gets completely overwritten from `('database',)` to `('security',)` without any warning or error.

2. **Breaks Tag-Based Check Filtering**: Django's check framework allows running specific subsets of checks by tag (e.g., `manage.py check --tag database`). When a check's tags are overwritten, it becomes unreachable via the original tags, breaking deployment scripts and CI/CD pipelines that rely on tag-based filtering.

3. **Violates Set Semantics**: The registry uses a `set()` to store checks (line 31: `self.registered_checks = set()`), which implies deduplication. However, the tag overwrite happens before the set operation (line 56 overwrites tags, line 62 adds to set), meaning the set prevents duplicate functions but not duplicate registrations with different tags.

4. **Common Real-World Scenarios**: This bug will occur in production when:
   - A check is imported and registered by multiple Django apps
   - Developers use multiple `@register()` decorators thinking tags accumulate
   - Third-party packages and the main application both register the same check
   - During refactoring when check registrations are moved between modules

5. **No Documentation of Behavior**: The docstring (lines 36-48) shows an example of registering with multiple tags in one call but doesn't mention what happens with duplicate registrations. Users reasonably expect either an error or tag accumulation, not silent overwrites.

## Relevant Context

The bug occurs in `/django/core/checks/registry.py` at line 56 where `check.tags = tags` unconditionally overwrites any existing tags. The check function is stored in a set (line 62: `checks.add(check)`), which prevents storing the same function twice, but by that point the tags have already been overwritten.

The `run_checks()` method (line 86) filters checks by comparing tags: `if not set(check.tags).isdisjoint(tags)`. Since the tags were overwritten, checks registered with the first tag will no longer match.

Django documentation on the check framework: https://docs.djangoproject.com/en/stable/topics/checks/
Source code: https://github.com/django/django/blob/main/django/core/checks/registry.py

## Proposed Fix

The most appropriate fix is to raise an explicit error when attempting duplicate registration, as this makes the issue immediately visible rather than causing silent failures:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -50,6 +50,16 @@ class CheckRegistry:

         def inner(check):
             if not func_accepts_kwargs(check):
                 raise TypeError(
                     "Check functions must accept keyword arguments (**kwargs)."
                 )
+
+            # Check if this function is already registered
+            checks = (
+                self.deployment_checks
+                if kwargs.get("deploy")
+                else self.registered_checks
+            )
+            if check in checks:
+                existing_tags = check.tags if hasattr(check, 'tags') else ()
+                raise ValueError(
+                    f"Check function {check.__name__!r} is already registered "
+                    f"with tags {existing_tags}. To register with multiple tags, "
+                    f"pass them all in one call: register(check, 'tag1', 'tag2')"
+                )
+
             check.tags = tags
-            checks = (
-                self.deployment_checks
-                if kwargs.get("deploy")
-                else self.registered_checks
-            )
             checks.add(check)
             return check
```