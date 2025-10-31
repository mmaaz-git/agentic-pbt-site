# Bug Report: Django truncate_name Length Limit Violation with Namespaces

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_name` function fails to respect the specified length limit when the identifier contains a namespace prefix (e.g., `SCHEMA"."TABLE`). The function only checks the table name portion's length, ignoring the namespace, causing the final truncated identifier to exceed the specified length limit.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from django.db.backends.utils import truncate_name


def calculate_identifier_length(identifier):
    stripped = identifier.strip('"')
    return len(stripped)


@given(
    namespace=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu',))),
    table_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu',))),
    length=st.integers(min_value=10, max_value=30)
)
@settings(max_examples=1000)
@example(namespace='SCHEMA', table_name='VERYLONGTABLENAME', length=20)
def test_truncate_name_respects_length_with_namespace(namespace, table_name, length):
    identifier = f'{namespace}"."{table_name}'
    result = truncate_name(identifier, length=length)
    result_length = calculate_identifier_length(result)

    assert result_length <= length
```

**Failing input**: `identifier='SCHEMA"."VERYLONGTABLENAME', length=20`

## Reproducing the Bug

```python
from django.db.backends.utils import truncate_name

identifier = 'SCHEMA"."VERYLONGTABLENAME'
length = 20

result = truncate_name(identifier, length=length)
print(f"Input: {identifier}")
print(f"Length limit: {length}")
print(f"Result: {result}")
print(f"Result length: {len(result.strip('\"'))}")

assert len(result.strip('"')) <= length
```

Output:
```
Input: SCHEMA"."VERYLONGTABLENAME
Length limit: 20
Result: SCHEMA"."VERYLONGTABLENAME
Result length: 25
AssertionError
```

The result exceeds the limit by 5 characters (25 > 20).

## Why This Is A Bug

The `truncate_name` function's docstring states: "Shorten an SQL identifier to a repeatable mangled version with the given length." This clearly indicates the entire identifier should be shortened to fit within the specified length.

However, the function only checks if the table name portion (without namespace) fits within the limit:

```python
if length is None or len(name) <= length:
    return identifier
```

When a namespace is present (e.g., `SCHEMA"."TABLE`), the function checks only `len("TABLE")` against the limit, ignoring the `SCHEMA"."` prefix. This causes the returned identifier to exceed the specified length.

This is particularly problematic for Oracle databases where identifiers have a 30-character limit. When `truncate_name` is called from `quote_name` with `max_name_length()` = 30, identifiers with namespaces can exceed this limit, potentially causing database errors.

## Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -290,7 +290,12 @@ def truncate_name(identifier, length=None, hash_len=4):
     """
     namespace, name = split_identifier(identifier)

-    if length is None or len(name) <= length:
+    if length is None:
+        return identifier
+
+    # Check the total identifier length, including namespace prefix if present
+    total_length = len(namespace) + 3 + len(name) if namespace else len(name)
+    if total_length <= length:
         return identifier

     digest = names_digest(name, length=hash_len)
+
+    # Account for namespace when calculating available space for the name
+    if namespace:
+        available_length = length - len(namespace) - 3 - hash_len
+        if available_length < 1:
+            # Namespace itself is too long, hash the namespace as well
+            namespace_hash = names_digest(namespace, length=hash_len)
+            available_length = length - hash_len - 3 - hash_len
+            if available_length < 1:
+                raise ValueError(f"Cannot truncate identifier to {length} characters")
+            return "%s%s%s" % (
+                namespace_hash + '"."',
+                name[:available_length],
+                names_digest(name, length=hash_len),
+            )
+    else:
+        available_length = length - hash_len
+
     return "%s%s%s" % (
         '%s"."' % namespace if namespace else "",
-        name[: length - hash_len],
+        name[:available_length],
         digest,
     )
```

Alternatively, a simpler fix that preserves the namespace but properly truncates the name:

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -290,11 +290,20 @@ def truncate_name(identifier, length=None, hash_len=4):
     """
     namespace, name = split_identifier(identifier)

-    if length is None or len(name) <= length:
+    if length is None:
+        return identifier
+
+    # Calculate the actual length needed
+    namespace_overhead = len(namespace) + 3 if namespace else 0
+    total_length = namespace_overhead + len(name)
+
+    if total_length <= length:
         return identifier
+
+    # Adjust available length for the name portion
+    available_name_length = length - namespace_overhead - hash_len

     digest = names_digest(name, length=hash_len)
     return "%s%s%s" % (
         '%s"."' % namespace if namespace else "",
-        name[: length - hash_len],
+        name[:available_name_length],
         digest,
     )
```