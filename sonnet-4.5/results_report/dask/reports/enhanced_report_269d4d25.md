# Bug Report: Django truncate_name Violates Length Limit with Namespaced Identifiers

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_name` function fails to respect the specified length limit when processing identifiers with namespace prefixes (e.g., `SCHEMA"."TABLE`), returning identifiers that exceed the maximum length parameter.

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


if __name__ == "__main__":
    test_truncate_name_respects_length_with_namespace()
```

<details>

<summary>
**Failing input**: `namespace='SCHEMA', table_name='VERYLONGTABLENAME', length=20`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 26, in <module>
    test_truncate_name_respects_length_with_namespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 11, in test_truncate_name_respects_length_with_namespace
    namespace=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu',))),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 22, in test_truncate_name_respects_length_with_namespace
    assert result_length <= length
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying explicit example: test_truncate_name_respects_length_with_namespace(
    namespace='SCHEMA',
    table_name='VERYLONGTABLENAME',
    length=20,
)
```
</details>

## Reproducing the Bug

```python
from django.db.backends.utils import truncate_name

# Test case from the bug report
identifier = 'SCHEMA"."VERYLONGTABLENAME'
length = 20

result = truncate_name(identifier, length=length)

print(f"Input: {identifier}")
print(f"Length limit: {length}")
print(f"Result: {result}")
print(f"Result length: {len(result.strip('\"'))}")
print()

# Check if the result exceeds the limit
stripped_result = result.strip('"')
if len(stripped_result) > length:
    print(f"ERROR: Result length ({len(stripped_result)}) exceeds limit ({length})")
    print(f"Excess characters: {len(stripped_result) - length}")
else:
    print(f"OK: Result length ({len(stripped_result)}) is within limit ({length})")

# Additional test cases
print("\n--- Additional Test Cases ---")

# Test 1: Without namespace - should truncate correctly
test1_id = "VERYLONGTABLENAME"
test1_result = truncate_name(test1_id, length=10)
print(f"\nTest 1 (no namespace):")
print(f"  Input: {test1_id}, Limit: 10")
print(f"  Result: {test1_result}")
print(f"  Result length: {len(test1_result.strip('\"'))}")

# Test 2: With namespace, very long table name
test2_id = 'SCHEMA"."VERYLONGTABLENAMETHATEXCEEDSLIMIT'
test2_result = truncate_name(test2_id, length=20)
print(f"\nTest 2 (namespace + very long table):")
print(f"  Input: {test2_id}, Limit: 20")
print(f"  Result: {test2_result}")
print(f"  Result length: {len(test2_result.strip('\"'))}")

# Test 3: Edge case with short namespace
test3_id = 'A"."BCDEFGHIJKLMNOPQRSTUVWXYZ'
test3_result = truncate_name(test3_id, length=15)
print(f"\nTest 3 (short namespace):")
print(f"  Input: {test3_id}, Limit: 15")
print(f"  Result: {test3_result}")
print(f"  Result length: {len(test3_result.strip('\"'))}")

# Assertion that should fail
print("\n--- Final Assertion ---")
try:
    assert len(result.strip('"')) <= length
    print("ASSERTION PASSED: Result is within length limit")
except AssertionError:
    print(f"ASSERTION FAILED: Result ({len(result.strip('\"'))}) exceeds limit ({length})")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AssertionError: Result exceeds specified length limit
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/repo.py", line 53, in <module>
    assert len(result.strip('"')) <= length
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Input: SCHEMA"."VERYLONGTABLENAME
Length limit: 20
Result: SCHEMA"."VERYLONGTABLENAME
Result length: 26

ERROR: Result length (26) exceeds limit (20)
Excess characters: 6

--- Additional Test Cases ---

Test 1 (no namespace):
  Input: VERYLONGTABLENAME, Limit: 10
  Result: VERYLO1d9f
  Result length: 10

Test 2 (namespace + very long table):
  Input: SCHEMA"."VERYLONGTABLENAMETHATEXCEEDSLIMIT, Limit: 20
  Result: SCHEMA"."VERYLONGTABLENAMd965
  Result length: 29

Test 3 (short namespace):
  Input: A"."BCDEFGHIJKLMNOPQRSTUVWXYZ, Limit: 15
  Result: A"."BCDEFGHIJKL5145
  Result length: 19

--- Final Assertion ---
ASSERTION FAILED: Result (26) exceeds limit (20)
```
</details>

## Why This Is A Bug

The `truncate_name` function's docstring explicitly states: "Shorten an SQL identifier to a repeatable mangled version with the given length." This contract is violated when the function returns identifiers exceeding the specified length parameter.

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/django/db/backends/utils.py:293-294` where the function only checks if the table name portion fits within the limit:

```python
if length is None or len(name) <= length:
    return identifier
```

When a namespace is present (e.g., `SCHEMA"."TABLE`), the function checks only `len("TABLE")` against the limit, ignoring the `SCHEMA"."` prefix (8 additional characters). In our test case:
- Table name: `VERYLONGTABLENAME` (17 characters)
- Check: 17 ≤ 20 (passes)
- Returned: `SCHEMA"."VERYLONGTABLENAME` (26 characters)
- Result: 26 > 20 (violates length contract)

This has real-world implications for Django applications using Oracle databases, which enforce a strict 30-character limit on identifiers. When `truncate_name` is called from `quote_name` with `max_name_length()=30`, namespaced identifiers can exceed this limit, potentially causing database errors in production.

## Relevant Context

- **Django Version**: 5.2.6
- **Function Location**: `django.db.backends.utils.truncate_name` (line 283-301)
- **Related Functions**: `split_identifier()` (line 269-280) splits namespace from table name
- **Database Impact**: Oracle backends rely on this function to ensure identifiers meet the 30-character limit
- **Docstring Intent**: While the docstring mentions "truncate the table portion only" for namespaced identifiers, this describes what gets truncated (table name), not that the length check should ignore the namespace

The bug affects all Django versions that include this implementation and manifests whenever:
1. An identifier contains a namespace prefix
2. The table name alone fits within the length limit
3. The complete identifier (namespace + separator + table) exceeds the limit

## Proposed Fix

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
+    # Calculate the total identifier length including namespace
+    namespace_overhead = len(namespace) + 3 if namespace else 0  # 3 chars for "."
+    total_length = namespace_overhead + len(name)
+
+    if total_length <= length:
         return identifier

     digest = names_digest(name, length=hash_len)
+    # Account for namespace when calculating available space for the truncated name
+    available_name_length = length - namespace_overhead - hash_len
+
     return "%s%s%s" % (
         '%s"."' % namespace if namespace else "",
-        name[: length - hash_len],
+        name[:available_name_length],
         digest,
     )
```