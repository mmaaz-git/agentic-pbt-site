# Bug Report: pandas.Categorical.remove_categories Unexpectedly Sorts Categories

**Target**: `pandas.core.arrays.Categorical.remove_categories`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `remove_categories` method on unordered Categorical objects unexpectedly sorts the remaining categories alphabetically, which is undocumented and inconsistent with other category manipulation methods and with ordered Categoricals.

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
```

**Failing input**: `cat = Categorical(['00'], categories=['00', '0'])`

## Reproducing the Bug

```python
import pandas as pd

cat = pd.Categorical(['c', 'b', 'a'], categories=['c', 'b', 'a', 'x'])
result = cat.remove_categories(['x'])

print(f'Original: {list(cat.categories)}')
print(f'After remove_categories: {list(result.categories)}')

assert list(result.categories) == ['c', 'b', 'a']
```

**Output:**
```
Original: ['c', 'b', 'a', 'x']
After remove_categories: ['a', 'b', 'c']
AssertionError
```

## Why This Is A Bug

1. **Undocumented behavior**: The docstring for `remove_categories` does not mention sorting
2. **Inconsistent with ordered categoricals**: For ordered Categoricals, `remove_categories` preserves order
3. **Inconsistent with similar methods**: `remove_unused_categories()`, `set_categories()`, and `add_categories()` all preserve order
4. **Breaks user expectations**: Category order can be semantically meaningful even for unordered categoricals (e.g., display order, domain-specific ordering like ["Low", "Medium", "High"])
5. **Breaking invariant**: `add_categories` followed by `remove_categories` should preserve the existing categories' order

## Fix

The issue appears to be in the `remove_categories` implementation for unordered categoricals. The fix should ensure that the remaining categories maintain their original order, similar to how ordered categoricals are handled.

Expected behavior:
```python
cat = pd.Categorical(['c'], categories=['c', 'b', 'a', 'x'])
result = cat.remove_categories(['x'])
assert list(result.categories) == ['c', 'b', 'a']  # Order preserved
```