# Bug Report: attrs.filters._split_what Generator Exhaustion Causes Silent Data Loss

**Target**: `attr.filters._split_what`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_split_what` function silently loses data when passed a generator instead of a list, because it iterates over the input three times without converting it to a reusable collection first, causing the second and third iterations to receive an exhausted generator.

## Property-Based Test

```python
from attr.filters import _split_what
from hypothesis import given, strategies as st

@given(st.lists(st.one_of(
    st.sampled_from([int, str, float]),
    st.text(min_size=1, max_size=20)
), min_size=1, max_size=20))
def test_split_what_generator_vs_list(items):
    gen_classes, gen_names, gen_attrs = _split_what(x for x in items)
    list_classes, list_names, list_attrs = _split_what(items)

    assert gen_classes == list_classes
    assert gen_names == list_names
    assert gen_attrs == list_attrs

if __name__ == "__main__":
    test_split_what_generator_vs_list()
```

<details>

<summary>
**Failing input**: `items=['0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 17, in <module>
    test_split_what_generator_vs_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 5, in test_split_what_generator_vs_list
    st.sampled_from([int, str, float]),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 13, in test_split_what_generator_vs_list
    assert gen_names == list_names
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_split_what_generator_vs_list(
    items=['0'],
)
```
</details>

## Reproducing the Bug

```python
from attr.filters import _split_what

# Test case demonstrating the bug
items = [int, str, "name1", "name2", float]

# Test with generator (bug occurs here)
gen = (x for x in items)
classes_gen, names_gen, attrs_gen = _split_what(gen)

print("=== Generator Input ===")
print(f"Classes: {classes_gen}")
print(f"Names: {names_gen}")
print(f"Attrs: {attrs_gen}")

# Test with list (correct behavior)
classes_list, names_list, attrs_list = _split_what(items)

print("\n=== List Input ===")
print(f"Classes: {classes_list}")
print(f"Names: {names_list}")
print(f"Attrs: {attrs_list}")

# Verify the bug
print("\n=== Assertion Checks ===")
try:
    assert classes_gen == frozenset({int, str, float})
    print("✓ Classes assertion passed")
except AssertionError:
    print("✗ Classes assertion failed")

try:
    assert names_gen == frozenset({"name1", "name2"})
    print("✓ Names assertion passed")
except AssertionError:
    print("✗ Names assertion failed - Expected {'name1', 'name2'}, got:", names_gen)

try:
    assert attrs_gen == frozenset()
    print("✓ Attrs assertion passed (no Attribute objects)")
except AssertionError:
    print("✗ Attrs assertion failed")
```

<details>

<summary>
Generator exhaustion causes empty names frozenset
</summary>
```
=== Generator Input ===
Classes: frozenset({<class 'float'>, <class 'int'>, <class 'str'>})
Names: frozenset()
Attrs: frozenset()

=== List Input ===
Classes: frozenset({<class 'float'>, <class 'int'>, <class 'str'>})
Names: frozenset({'name2', 'name1'})
Attrs: frozenset()

=== Assertion Checks ===
✓ Classes assertion passed
✗ Names assertion failed - Expected {'name1', 'name2'}, got: frozenset()
✓ Attrs assertion passed (no Attribute objects)
```
</details>

## Why This Is A Bug

This violates expected behavior because the function returns incorrect results without any warning or error when given a generator. The function's docstring states "Returns a tuple of `frozenset`s of classes and attributes" but doesn't specify that the input must be reusable. The implementation at lines 14-18 in `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/filters.py` creates three generator comprehensions that each iterate over `what`:

1. First comprehension (line 15): Filters for types - exhausts the generator
2. Second comprehension (line 16): Filters for strings - receives exhausted generator, returns empty frozenset
3. Third comprehension (line 17): Filters for Attribute objects - receives exhausted generator, returns empty frozenset

This causes silent data loss - string names and Attribute objects are never detected when the input is a generator, even though they are present in the original data. No exception is raised, making this bug difficult to detect.

## Relevant Context

- **Private Function**: `_split_what` is marked as private with a leading underscore, indicating it's an internal implementation detail
- **Public API Unaffected**: The public functions `include()` and `exclude()` (lines 21-45 and 48-72) work correctly because they pass `*what` arguments as tuples to `_split_what`, never generators
- **Documentation**: The function has minimal documentation that doesn't specify input requirements or warn about multiple iterations
- **Python Convention**: Generators are standard Python iterables that users might naturally pass to functions accepting iterables
- **Use Cases**: Direct calls to `_split_what` would be unusual but possible since the function is importable from `attr.filters`

Source code location: `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/filters.py`

## Proposed Fix

```diff
--- a/attr/filters.py
+++ b/attr/filters.py
@@ -10,6 +10,7 @@ from ._make import Attribute
 def _split_what(what):
     """
     Returns a tuple of `frozenset`s of classes and attributes.
     """
+    what = tuple(what)
     return (
         frozenset(cls for cls in what if isinstance(cls, type)),
         frozenset(cls for cls in what if isinstance(cls, str)),
```