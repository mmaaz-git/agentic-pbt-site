# Bug Report: pandas.Categorical.remove_categories Unexpectedly Sorts Categories

**Target**: `pandas.core.arrays.Categorical.remove_categories`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `remove_categories` method on unordered Categorical objects unexpectedly sorts the remaining categories alphabetically, while ordered Categoricals preserve the original order. This undocumented inconsistency violates user expectations and breaks the invariant that adding and removing the same category should preserve the original order.

## Property-Based Test

```python
import pandas.core.arrays as arrays
from hypothesis import given, settings, strategies as st, assume

def categoricals():
    @st.composite
    def _categoricals(draw):
        categories = draw(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10, unique=True))
        codes = draw(st.lists(st.integers(min_value=-1, max_value=len(categories)-1), min_size=1, max_size=50))
        return arrays.Categorical.from_codes(codes, categories=categories)
    return _categoricals()

@given(categoricals())
@settings(max_examples=200)
def test_categorical_add_remove_categories_identity(cat):
    original_categories = list(cat.categories)
    new_cat = 'NEW_CATEGORY_XYZ'
    assume(new_cat not in original_categories)

    cat_added = cat.add_categories([new_cat])
    cat_removed = cat_added.remove_categories([new_cat])

    assert list(cat_removed.categories) == original_categories

if __name__ == "__main__":
    test_categorical_add_remove_categories_identity()
```

<details>

<summary>
**Failing input**: `cat = Categorical(['00'], categories=['00', '0'])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 25, in <module>
    test_categorical_add_remove_categories_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 13, in test_categorical_add_remove_categories_identity
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 22, in test_categorical_add_remove_categories_identity
    assert list(cat_removed.categories) == original_categories
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_categorical_add_remove_categories_identity(
    cat=['00']
    Categories (2, object): ['00', '0'],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Simple example demonstrating the bug
cat = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result = cat.remove_categories(['x'])

print(f'Original categories: {list(cat.categories)}')
print(f'After remove_categories: {list(result.categories)}')

# This assertion will fail - categories get sorted alphabetically
assert list(result.categories) == ['c', 'b', 'a'], f"Expected ['c', 'b', 'a'] but got {list(result.categories)}"
```

<details>

<summary>
AssertionError: Categories unexpectedly sorted from ['c', 'b', 'a'] to ['a', 'b', 'c']
</summary>
```
Original categories: ['c', 'b', 'a', 'x']
After remove_categories: ['a', 'b', 'c']
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/repo.py", line 11, in <module>
    assert list(result.categories) == ['c', 'b', 'a'], f"Expected ['c', 'b', 'a'] but got {list(result.categories)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected ['c', 'b', 'a'] but got ['a', 'b', 'c']
```
</details>

## Why This Is A Bug

1. **Undocumented behavior**: The docstring for `remove_categories` states it "Remove the specified categories" with no mention of sorting. The example in the docstring happens to show a case where the alphabetical order matches the original order, masking this behavior.

2. **Inconsistent with ordered categoricals**: The same operation on ordered Categoricals preserves the original order. The code explicitly uses `difference(removals, sort=False)` for ordered but just `difference(removals)` for unordered, causing this inconsistency.

3. **Violates the add/remove invariant**: Adding a category and then removing it should be a no-op for the existing categories, but currently it sorts them.

4. **Inconsistent with all other category methods**: Methods like `add_categories()`, `remove_unused_categories()`, and `set_categories()` all preserve the original category order for unordered categoricals.

5. **Breaks legitimate use cases**: Even for unordered categoricals, the order can be semantically meaningful (e.g., display order, domain-specific ordering like ["Low", "Medium", "High"], or maintaining consistency across data processing pipelines).

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/categorical.py` lines 1400-1404:

```python
new_categories = (
    self.dtype.categories.difference(removals, sort=False)
    if self.dtype.ordered is True
    else self.dtype.categories.difference(removals)
)
```

The `pandas.Index.difference()` method defaults to `sort=None` which attempts to sort the result, while `sort=False` preserves the original order. This explains why ordered categoricals maintain order (explicit `sort=False`) while unordered ones get sorted (default behavior).

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.Categorical.remove_categories.html

Related issue discussions may exist around maintaining category order in pandas GitHub issues.

## Proposed Fix

The fix is straightforward - use `sort=False` for both ordered and unordered categoricals to maintain consistency:

```diff
--- a/pandas/core/arrays/categorical.py
+++ b/pandas/core/arrays/categorical.py
@@ -1398,10 +1398,7 @@ class Categorical(NDArrayBacked[Ordered], PandasObject, ObjectStringArrayMixin)

         removals = Index(removals).unique().dropna()
-        new_categories = (
-            self.dtype.categories.difference(removals, sort=False)
-            if self.dtype.ordered is True
-            else self.dtype.categories.difference(removals)
-        )
+        new_categories = self.dtype.categories.difference(removals, sort=False)
         not_included = removals.difference(self.dtype.categories)

         if len(not_included) != 0:
```