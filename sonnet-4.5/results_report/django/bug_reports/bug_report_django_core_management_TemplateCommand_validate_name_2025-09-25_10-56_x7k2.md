# Bug Report: Django TemplateCommand validate_name ValueError with __main__

**Target**: `django.core.management.templates.TemplateCommand.validate_name`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The validate_name method crashes with ValueError when validating the name '__main__' because find_spec('__main__') raises ValueError instead of returning None or a valid spec.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.management.templates import TemplateCommand
from django.core.management.base import CommandError
import pytest


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20))
def test_validate_name_no_unexpected_errors(name):
    """
    Property: validate_name should only raise CommandError for invalid names,
    never ValueError or other unexpected exceptions.
    """
    cmd = TemplateCommand()
    cmd.app_or_project = 'app'
    cmd.a_or_an = 'an'

    try:
        cmd.validate_name(name, 'name')
    except ValueError as e:
        # This is a bug - should only raise CommandError
        pytest.fail(f"Unexpected ValueError for name '{name}': {e}")
    except CommandError:
        # Expected validation failure
        pass
```

**Failing input**: `name = '__main__'`

## Reproducing the Bug

```python
from django.core.management.templates import TemplateCommand

cmd = TemplateCommand()
cmd.app_or_project = 'app'
cmd.a_or_an = 'an'

cmd.validate_name('__main__', 'name')
```

Output:
```
ValueError: __main__.__spec__ is None
```

## Why This Is A Bug

The validate_name method uses `find_spec(name)` to check if a name conflicts with an existing Python module:

```python
if find_spec(name) is not None:
    raise CommandError(...)
```

However, `find_spec('__main__')` raises `ValueError` because the `__main__` module has `__spec__ = None`, and find_spec doesn't handle this case gracefully.

This violates the method's contract - it should only raise `CommandError` for validation failures, not propagate unexpected exceptions like `ValueError`.

While `__main__` is an unlikely app/project name in practice (since it's not a valid identifier by convention), the crash represents a violation of the API contract. Any validation method should handle edge cases gracefully and not leak implementation details like `ValueError` to callers.

## Fix

Wrap the find_spec call in a try-except to handle ValueError:

```diff
def validate_name(self, name, name_or_dir="name"):
    if name is None:
        raise CommandError(...)
    # Check it's a valid directory name.
    if not name.isidentifier():
        raise CommandError(...)
    # Check that __spec__ doesn't exist.
-   if find_spec(name) is not None:
+   try:
+       spec = find_spec(name)
+   except ValueError:
+       # Some module names like '__main__' cause find_spec to raise ValueError
+       # Treat this as if the module exists (since it's a special Python name)
+       spec = True
+
+   if spec is not None:
        raise CommandError(...)
```