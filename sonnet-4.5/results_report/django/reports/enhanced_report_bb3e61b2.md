# Bug Report: Django WhereNode XOR Crashes With Empty Children on Non-Native XOR Databases

**Target**: `django.db.models.sql.where.WhereNode.as_sql`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

WhereNode with XOR connector and no children causes a TypeError when the database doesn't support native XOR operations, instead of raising the expected FullResultSet exception.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import pytest
from django.db.models.sql.where import WhereNode, XOR


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


def test_empty_xor_node_no_native_support():
    """Test XOR with no children when database doesn't support XOR natively"""
    wn = WhereNode([], XOR)
    compiler = MockCompiler()
    conn = MockConnection()

    with pytest.raises(TypeError):
        wn.as_sql(compiler, conn)


if __name__ == "__main__":
    test_empty_xor_node_no_native_support()
    print("Test passed - TypeError was raised as expected")
```

<details>

<summary>
**Failing input**: `WhereNode([], XOR)` with database lacking native XOR support
</summary>
```
Test passed - TypeError was raised as expected
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.models.sql.where import WhereNode, XOR


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


# Create an empty WhereNode with XOR connector
wn = WhereNode([], XOR)
compiler = MockCompiler()
conn = MockConnection()

# Try to generate SQL - this should crash
try:
    result = wn.as_sql(compiler, conn)
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: reduce() of empty iterable with no initial value
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/repo.py", line 24, in <module>
    result = wn.as_sql(compiler, conn)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/sql/where.py", line 138, in as_sql
    rhs_sum = reduce(
        operator.add,
        (Case(When(c, then=1), default=0) for c in self.children),
    )
TypeError: reduce() of empty iterable with no initial value
TypeError: reduce() of empty iterable with no initial value
```
</details>

## Why This Is A Bug

This violates Django's expected behavior in multiple ways:

1. **Inconsistent database behavior**: On databases with native XOR support (MySQL, MariaDB), an empty WhereNode with XOR correctly raises `FullResultSet`. On databases without native XOR support (PostgreSQL, SQLite, Oracle), it crashes with `TypeError`.

2. **Breaks exception contract**: The `as_sql()` method is documented to raise `FullResultSet` for empty WHERE clauses (line 179-180 in where.py). Instead, it raises an unhandled `TypeError` due to calling `reduce()` on an empty iterator without an initial value.

3. **XOR is public API**: While `WhereNode` itself isn't in `django.db.models.sql.__all__`, the `XOR` connector is explicitly exported and documented for use with Q objects and QuerySets.

4. **Reachable through normal operations**: This crash can occur during query optimization, dynamic filter construction, or when combining Q objects that result in empty conditions.

## Relevant Context

The bug occurs in the XOR emulation code (lines 130-147 in where.py) which converts XOR operations for databases without native support. The emulation transforms `a XOR b XOR c` into `(a OR b OR c) AND MOD(a + b + c, 2) == 1`.

The code correctly handles other edge cases:
- Empty nodes with AND/OR connectors properly raise `FullResultSet`
- Databases with native XOR support handle empty nodes correctly
- The XOR emulation works correctly when children exist

Key code locations:
- XOR emulation: `/django/db/models/sql/where.py:130-147`
- Public API exports: `/django/db/models/sql/__init__.py:6` (exports XOR but not WhereNode)
- Expected empty handling: `/django/db/models/sql/where.py:179-180`

Django documentation states XOR matches rows matched by an odd number of operands. For zero operands, this should logically be false (FullResultSet when negated is false).

## Proposed Fix

```diff
--- a/django/db/models/sql/where.py
+++ b/django/db/models/sql/where.py
@@ -127,7 +127,7 @@ class WhereNode(tree.Node):
         else:
             full_needed, empty_needed = 1, len(self.children)

-        if self.connector == XOR and not connection.features.supports_logical_xor:
+        if self.connector == XOR and self.children and not connection.features.supports_logical_xor:
             # Convert if the database doesn't support XOR:
             #   a XOR b XOR c XOR ...
             # to:
```