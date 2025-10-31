# Bug Report: django.core.checks.CheckRegistry Double Registration Overwrites Tags

**Target**: `django.core.checks.registry.CheckRegistry.register`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a check function is registered multiple times with different tags using `CheckRegistry.register()`, the second registration overwrites the tags from the first registration instead of accumulating them, making the check inaccessible via its original tags.

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


if __name__ == "__main__":
    # Run the test with Hypothesis
    test_double_registration_preserves_both_tags()
```

<details>

<summary>
**Failing input**: `tag1='0', tag2='N'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 27, in <module>
    test_double_registration_preserves_both_tags()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 7, in test_double_registration_preserves_both_tags
    def test_double_registration_preserves_both_tags(tag1, tag2):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 21, in test_double_registration_preserves_both_tags
    assert tag1 in available_tags, f"Tag '{tag1}' should still be available"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Tag '0' should still be available
Falsifying example: test_double_registration_preserves_both_tags(
    tag1='0',
    tag2='N',
)
```
</details>

## Reproducing the Bug

```python
from django.core.checks.registry import CheckRegistry
from django.core.checks import Info

registry = CheckRegistry()

def my_check(app_configs=None, **kwargs):
    return [Info("My check")]

# First registration with tag1
registry.register(my_check, "tag1")
print(f"After first registration - tags: {my_check.tags}")

# Second registration with tag2
registry.register(my_check, "tag2")
print(f"After second registration - tags: {my_check.tags}")

# Check available tags
available_tags = registry.tags_available()
print(f"Available tags: {available_tags}")

# Try to run checks with each tag
checks_tag1 = registry.run_checks(tags=["tag1"])
checks_tag2 = registry.run_checks(tags=["tag2"])

print(f"Checks with tag1: {len(checks_tag1)}")
print(f"Checks with tag2: {len(checks_tag2)}")
```

<details>

<summary>
Output showing tag1 is lost after second registration
</summary>
```
After first registration - tags: ('tag1',)
After second registration - tags: ('tag2',)
Available tags: {'tag2'}
Checks with tag1: 0
Checks with tag2: 1
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Silent data loss**: The first tag is completely lost without any warning or error. Users who registered a check with 'tag1' can no longer run it using that tag.

2. **Inconsistent with set behavior**: The implementation stores checks in a set (line 62 of registry.py: `checks.add(check)`), which correctly prevents duplicate functions. However, the tags attribute is overwritten (line 56: `check.tags = tags`), creating an inconsistency where the check object remains unique but its tags are replaced.

3. **Documentation doesn't warn against this pattern**: The Django documentation and docstring examples show registering with multiple tags in a single call, but don't explicitly state that multiple registrations will overwrite previous tags. The docstring at lines 40-48 shows the correct usage pattern but doesn't warn against multiple registrations.

4. **Violates principle of least surprise**: Users might reasonably expect that registering the same check multiple times would either:
   - Accumulate the tags (most intuitive)
   - Raise an error about duplicate registration
   - Keep the first registration and ignore the second

   Instead, it silently overwrites the first registration's tags while keeping the same function object in the set.

## Relevant Context

The issue occurs in `/django/core/checks/registry.py` at the `register` method (lines 34-70). The problematic behavior stems from:

- Line 56: `check.tags = tags` - directly overwrites the tags attribute
- Line 62: `checks.add(check)` - adds to a set, which has no effect on the second registration since the same function object is already in the set

The correct way to register a check with multiple tags (as shown in the documentation) is:
```python
registry.register(my_check, 'tag1', 'tag2')  # All tags in one call
```

This works as expected and makes the check available via both tags. However, the sequential registration pattern fails silently, which can lead to hard-to-debug issues where checks seem to disappear.

## Proposed Fix

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