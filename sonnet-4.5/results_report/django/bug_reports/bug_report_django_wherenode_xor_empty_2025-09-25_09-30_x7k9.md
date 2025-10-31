# Bug Report: WhereNode XOR Emulation Crashes on Empty Children

**Target**: `django.db.models.sql.where.WhereNode.as_sql`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a `WhereNode` with `XOR` connector has no children and the database doesn't support native XOR operations, calling `as_sql()` raises a `TypeError` due to an empty `reduce()` call without an initial value.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

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
```

**Failing input**: `WhereNode([], XOR)` on a database without native XOR support

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models.sql.where import WhereNode, XOR


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


wn = WhereNode([], XOR)
compiler = MockCompiler()
conn = MockConnection()

wn.as_sql(compiler, conn)
```

This raises:
```
TypeError: reduce() of empty iterable with no initial value
```

## Why This Is A Bug

`WhereNode` is part of Django's public API (exported in `django.db.models.sql.__all__`), and `XOR` is a documented connector type. While uncommon, an empty `WhereNode` can occur during query optimization or dynamic filter construction.

The XOR emulation code at lines 130-147 in `where.py` attempts to transform XOR into equivalent SQL for databases that don't support it natively. However, it uses `reduce()` without an initial value (lines 138-140), which fails when `self.children` is empty.

The code handles other edge cases (empty result sets, full result sets) but fails to check for empty children before the XOR emulation, leading to an unhandled crash instead of the expected `FullResultSet` exception.

## Fix

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

The fix adds a check for `self.children` before attempting XOR emulation, allowing the existing empty-handling logic (line 179) to correctly raise `FullResultSet` for empty XOR nodes.