# Bug Report: django.db.models.functions.Collate - Accepts Invalid Collation Names with Trailing Newlines

**Target**: `django.db.models.functions.Collate`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Collate` function's validation regex incorrectly accepts collation names containing trailing newlines due to Python's `$` anchor matching before a trailing newline, violating the intended validation contract.

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

if __name__ == "__main__":
    test_collate_should_reject_trailing_newline()
```

<details>

<summary>
**Failing input**: `'0\n'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 18, in <module>
    test_collate_should_reject_trailing_newline()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 10, in test_collate_should_reject_trailing_newline
    def test_collate_should_reject_trailing_newline(collation_with_newline):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 13, in test_collate_should_reject_trailing_newline
    assert False, f"Collate should reject {repr(collation_with_newline)} but it didn't"
           ^^^^^
AssertionError: Collate should reject '0\n' but it didn't
Falsifying example: test_collate_should_reject_trailing_newline(
    collation_with_newline='0\n',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models.functions import Collate
from django.db.models.expressions import Value

# Test case that should fail validation but doesn't
collation_with_newline = "utf8_general_ci\n"

try:
    result = Collate(Value("test"), collation_with_newline)
    print(f"ERROR: Collate accepted {repr(collation_with_newline)} when it should have rejected it")
    print(f"Stored collation: {repr(result.collation)}")
except ValueError as e:
    print(f"Correctly rejected {repr(collation_with_newline)} with error: {e}")
```

<details>

<summary>
Collate incorrectly accepts collation name with trailing newline
</summary>
```
ERROR: Collate accepted 'utf8_general_ci\n' when it should have rejected it
Stored collation: 'utf8_general_ci\n'
```
</details>

## Why This Is A Bug

The regex pattern `^[\w-]+$` at line 109 of `/home/npc/miniconda/lib/python3.13/site-packages/django/db/models/functions/comparison.py` is intended to validate that collation names contain only word characters (letters, digits, underscores) and hyphens. However, in Python regular expressions, the `$` anchor matches either at the end of the string OR before a newline at the end of the string. This means strings like `"utf8_general_ci\n"` or even `"0\n"` are incorrectly accepted as valid collation names.

This violates the validation contract because:
1. The code comment at lines 107-108 references PostgreSQL identifier rules, which explicitly disallow whitespace in unquoted identifiers
2. No legitimate database collation name contains newlines - typical examples are "utf8_general_ci", "latin1_swedish_ci", "C", "POSIX", "en_US.utf8"
3. The validation is meant to prevent SQL injection and ensure well-formed SQL queries
4. Accepting collation names with embedded newlines could lead to malformed SQL when the collation is inserted into SQL templates

## Relevant Context

The Collate class is defined in Django's database functions module and is used to specify a collation for string comparisons in database queries. The validation regex at line 109 is designed to ensure only safe, valid collation names are accepted.

The bug occurs because Python's regex `$` anchor has special behavior - it matches before a trailing newline, not just at the absolute end of the string. This is documented Python behavior but often catches developers by surprise.

Relevant code location: `/home/npc/miniconda/lib/python3.13/site-packages/django/db/models/functions/comparison.py:109`

PostgreSQL identifier documentation referenced in the code: https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS

## Proposed Fix

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