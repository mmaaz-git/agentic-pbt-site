# Bug Report: django.db.models.functions.Collate - Accepts Invalid Collation Names with Trailing Newlines

**Target**: `django.db.models.functions.Collate`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Collate` function's validation regex incorrectly accepts collation names containing trailing newlines due to Python's `$` anchor matching before a trailing newline. This violates the intended validation contract and could lead to malformed SQL queries.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.db.models.functions import Collate
from django.db.models.expressions import Value


@given(st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd']), min_size=1).map(lambda x: x + '\n'))
def test_collate_should_reject_trailing_newline(collation_with_newline):
    try:
        Collate(Value("test"), collation_with_newline)
        assert False, f"Collate should reject {repr(collation_with_newline)} but it didn't"
    except ValueError:
        pass
```

**Failing input**: `'0\n'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models.functions import Collate
from django.db.models.expressions import Value

collation_with_newline = "utf8_general_ci\n"

result = Collate(Value("test"), collation_with_newline)
print(f"Stored collation: {repr(result.collation)}")
```

## Why This Is A Bug

The regex pattern `^[\w-]+$` is intended to validate that collation names contain only word characters and hyphens. However, Python's `$` anchor matches before a trailing newline, not just at the absolute end of the string. This means inputs like `"utf8_general_ci\n"` are incorrectly accepted.

The validation is meant to prevent SQL injection and ensure well-formed SQL queries. Accepting collation names with embedded newlines violates this security boundary and could lead to malformed queries or unexpected behavior when the collation name is inserted into SQL templates.

## Fix

Replace `$` with `\Z` in the regex pattern. The `\Z` anchor only matches at the absolute end of the string, not before a trailing newline.

```diff
--- a/django/db/models/functions/comparison.py
+++ b/django/db/models/functions/comparison.py
@@ -106,7 +106,7 @@ class Collate(Func):
     allowed_default = False
     # Inspired from
     # https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
-    collation_re = _lazy_re_compile(r"^[\w-]+$")
+    collation_re = _lazy_re_compile(r"^[\w-]+\Z")

     def __init__(self, expression, collation):
         if not (collation and self.collation_re.match(collation)):
```