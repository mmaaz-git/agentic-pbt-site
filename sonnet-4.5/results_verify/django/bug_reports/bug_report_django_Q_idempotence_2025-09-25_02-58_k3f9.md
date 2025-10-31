# Bug Report: django.db.models.Q Idempotence Violation

**Target**: `django.db.models.Q`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Q objects violate the idempotence property: combining a Q object with itself using `&` or `|` should return an equivalent object, but instead creates redundant conditions in the generated SQL, making queries less efficient.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.db.models import Q


def q_objects():
    return st.builds(Q, x=st.integers()) | st.builds(Q, name=st.text())


@given(q_objects())
@settings(max_examples=500)
def test_q_and_idempotent(q):
    assert (q & q) == q


@given(q_objects())
@settings(max_examples=500)
def test_q_or_idempotent(q):
    assert (q | q) == q
```

**Failing input**: `Q(x=0)`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

import django
from django.conf import settings

settings.configure(
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
    INSTALLED_APPS=['django.contrib.contenttypes'],
)
django.setup()

from django.db import models
from django.db.models import Q


class TestModel(models.Model):
    x = models.IntegerField()
    class Meta:
        app_label = 'test_app'


q = Q(x=0)
q_and = q & q
q_or = q | q

print(f"q: {q}")
print(f"q & q: {q_and}")
print(f"q | q: {q_or}")
print(f"q == (q & q): {q == q_and}")
print(f"q == (q | q): {q == q_or}")

print(f"\nSQL for q: {TestModel.objects.filter(q).query}")
print(f"SQL for q & q: {TestModel.objects.filter(q_and).query}")
print(f"SQL for q | q: {TestModel.objects.filter(q_or).query}")
```

**Output:**
```
q: (AND: ('x', 0))
q & q: (AND: ('x', 0), ('x', 0))
q | q: (OR: ('x', 0), ('x', 0))
q == (q & q): False
q == (q | q): False

SQL for q: WHERE "test_app_testmodel"."x" = 0
SQL for q & q: WHERE ("test_app_testmodel"."x" = 0 AND "test_app_testmodel"."x" = 0)
SQL for q | q: WHERE ("test_app_testmodel"."x" = 0 OR "test_app_testmodel"."x" = 0)
```

## Why This Is A Bug

In boolean algebra, idempotence is a fundamental property: `x ∧ x = x` and `x ∨ x = x`. Django's Q objects claim to "combine filters logically", yet they violate this basic property.

This leads to:
1. **Inefficient SQL**: Redundant conditions like `WHERE (x = 0 AND x = 0)` instead of `WHERE x = 0`
2. **Equality violations**: `q & q != q` breaks code that compares Q objects
3. **Caching issues**: Deduplication logic may fail to recognize equivalent queries

## Fix

The `_combine` method in `Q` should detect when combining a Q object with itself and return a copy instead of creating redundant children:

```diff
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -40,6 +40,8 @@ class Q(tree.Node):
         if not other and isinstance(other, Q):
             return self.copy()

+        if self == other:
+            return self.copy()
+
         obj = self.create(connector=conn)
         obj.add(self, conn)
         obj.add(other, conn)
```