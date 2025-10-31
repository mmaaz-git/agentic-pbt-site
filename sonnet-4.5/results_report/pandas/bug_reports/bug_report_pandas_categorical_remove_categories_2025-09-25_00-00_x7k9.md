# Bug Report: pandas.Categorical.remove_categories Sorts Unordered Categories

**Target**: `pandas.core.arrays.Categorical.remove_categories`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `remove_categories()` method incorrectly sorts the remaining categories for unordered Categoricals, while preserving order for ordered Categoricals. This behavior is backwards and violates the expected invariant that removing categories should only remove them, not reorder the remaining ones.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=5), min_size=2, max_size=20),
       st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=5), min_size=1, max_size=10))
def test_categorical_add_remove_roundtrip(initial_categories, new_categories):
    assume(len(set(initial_categories)) == len(initial_categories))
    assume(len(set(new_categories)) == len(new_categories))
    assume(not any(cat in initial_categories for cat in new_categories))

    cat = pd.Categorical(['a'], categories=initial_categories)
    original_categories = list(cat.categories)

    cat_with_added = cat.add_categories(new_categories)
    cat_removed = cat_with_added.remove_categories(new_categories)

    assert list(cat_removed.categories) == original_categories
```

**Failing input**: `initial_categories=['z', 'y', 'x']`, `new_categories=['a']`

## Reproducing the Bug

```python
import pandas as pd

cat = pd.Categorical([], categories=['z', 'y', 'x'])
print(f"Initial categories: {list(cat.categories)}")

cat_removed = cat.remove_categories(['x'])
print(f"After remove_categories(['x']): {list(cat_removed.categories)}")
```

**Output:**
```
Initial categories: ['z', 'y', 'x']
After remove_categories(['x']): ['y', 'z']
```

Expected: `['z', 'y']`
Actual: `['y', 'z']`

## Why This Is A Bug

1. **Violates preservation of insertion order**: The documentation states that `remove_categories` removes specified categories, with no mention of reordering the remaining ones.

2. **Inconsistent with ordered categoricals**: When `ordered=True`, the method correctly preserves category order. When `ordered=False` (the default), it incorrectly sorts them.

3. **Breaks semantic ordering**: Even for unordered categoricals, the category order can be semantically meaningful (e.g., database insertion order, user preference, etc.). The `ordered` flag only indicates whether comparison operations should respect the order, not whether the order should be destroyed.

4. **Violates round-trip property**: `cat.add_categories(x).remove_categories(x)` should return a categorical equivalent to the original.

## Fix

The bug is in `pandas/core/arrays/categorical.py` in the `remove_categories` method. The issue is this code:

```python
new_categories = (
    self.dtype.categories.difference(removals, sort=False)
    if self.dtype.ordered is True
    else self.dtype.categories.difference(removals)  # sort=True by default!
)
```

The fix is to always use `sort=False`:

```diff
--- a/pandas/core/arrays/categorical.py
+++ b/pandas/core/arrays/categorical.py
@@ -1682,10 +1682,7 @@ class Categorical(OpsMixin, PandasObject):
         if not is_list_like(removals):
             removals = [removals]

         removals = Index(removals).unique().dropna()
-        new_categories = (
-            self.dtype.categories.difference(removals, sort=False)
-            if self.dtype.ordered is True
-            else self.dtype.categories.difference(removals)
-        )
+        new_categories = self.dtype.categories.difference(removals, sort=False)
         not_included = removals.difference(self.dtype.categories)
```