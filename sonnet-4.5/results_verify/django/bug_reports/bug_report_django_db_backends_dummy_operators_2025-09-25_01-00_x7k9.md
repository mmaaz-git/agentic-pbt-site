# Bug Report: django.db.backends.dummy Shared Mutable `operators` Dict

**Target**: `django.db.backends.dummy.base.DatabaseWrapper`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DatabaseWrapper` class in `django.db.backends.dummy` uses a class-level mutable dict `operators = {}`, which is shared between all instances. Modifying the `operators` dict on one instance affects all other instances, violating instance isolation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import pytest
from hypothesis import given, strategies as st
from django.db.backends.dummy.base import DatabaseWrapper


@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.text())))
def test_operators_identity_differs_between_instances(settings):
    wrapper1 = DatabaseWrapper(settings)
    wrapper2 = DatabaseWrapper(settings)

    assert (
        wrapper1.operators is not wrapper2.operators
    ), "operators dict is the same object for different instances"
```

**Failing input**: `settings={}` (or any settings dict)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.backends.dummy.base import DatabaseWrapper

wrapper1 = DatabaseWrapper({})
wrapper2 = DatabaseWrapper({})

print(f"Same object? {wrapper1.operators is wrapper2.operators}")

wrapper1.operators['CUSTOM'] = 'custom_value'

print(f"wrapper1.operators: {wrapper1.operators}")
print(f"wrapper2.operators: {wrapper2.operators}")

assert 'CUSTOM' in wrapper2.operators, "Bug: operators dict is shared!"
```

## Why This Is A Bug

The `operators` class attribute is defined as `operators = {}` at the class level. In Python, mutable objects assigned at class level are shared among all instances. This means:

1. All `DatabaseWrapper` instances share the same `operators` dict object
2. Modifying `operators` on one instance affects all other instances
3. This violates the principle of instance isolation

While the dummy backend is not used in production, this could cause:
- Test pollution where one test modifying operators affects others
- Unexpected behavior in development/testing environments
- Confusion for developers who expect instance isolation

## Fix

Replace the class-level mutable dict with an instance-level dict in `__init__()`, or use a class method/property to ensure each instance gets its own copy:

```diff
--- a/django/db/backends/dummy/base.py
+++ b/django/db/backends/dummy/base.py
@@ -51,7 +51,6 @@ class DatabaseIntrospection(BaseDatabaseIntrospection):


 class DatabaseWrapper(BaseDatabaseWrapper):
-    operators = {}
     # Override the base class implementations with null
     # implementations. Anything that tries to actually
     # do something raises complain; anything that tries
@@ -71,6 +70,10 @@ class DatabaseWrapper(BaseDatabaseWrapper):
     introspection_class = DatabaseIntrospection
     ops_class = DatabaseOperations

+    def __init__(self, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+        self.operators = {}
+
     def is_usable(self):
         return True
```