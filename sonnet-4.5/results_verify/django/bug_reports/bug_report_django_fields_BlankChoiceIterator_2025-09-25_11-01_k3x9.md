# Bug Report: django.db.models.fields.BlankChoiceIterator Unpacks Tuple

**Target**: `django.db.models.fields.BlankChoiceIterator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`BlankChoiceIterator.__iter__` incorrectly unpacks the `blank_choice` tuple using `yield from`, resulting in individual tuple elements being yielded instead of the tuple itself. This breaks the expected structure where all choice items should be (value, label) tuples.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import django.db.models.fields as fields

choice_pair = st.tuples(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=20)
)

@given(
    st.lists(choice_pair, min_size=1, max_size=10),
    choice_pair
)
def test_blankchoiceiterator_preserves_tuple_structure(choices, blank_choice):
    bci = fields.BlankChoiceIterator(choices, blank_choice)
    result = list(bci)

    assert len(result) == len(choices) + 1, f"Expected {len(choices) + 1} choices, got {len(result)}"
    assert result[0] == blank_choice, f"Expected first element to be {blank_choice}, got {result[0]}"

    for i, choice in enumerate(choices):
        assert result[i + 1] == choice, f"Expected element {i+1} to be {choice}, got {result[i + 1]}"
```

**Failing input**: `choices=[('0', '0')], blank_choice=('0', '0')`

## Reproducing the Bug

```python
import django.db.models.fields as fields

choices = [('a', 'A'), ('b', 'B')]
blank_choice = ('', 'Empty')
bci = fields.BlankChoiceIterator(choices, blank_choice)

result = list(bci)

print("Expected:", [('', 'Empty'), ('a', 'A'), ('b', 'B')])
print("Actual:", result)
```

Output:
```
Expected: [('', 'Empty'), ('a', 'A'), ('b', 'B')]
Actual: ['', 'Empty', ('a', 'A'), ('b', 'B')]
```

## Why This Is A Bug

The `BlankChoiceIterator` class is designed to inject a blank choice at the beginning of a choices list for form fields. All choice items should be tuples of `(value, label)` format. However, the current implementation uses `yield from self.blank_choice`, which unpacks the tuple and yields its elements individually.

This violates the expected structure where every choice should be a 2-tuple. Code consuming this iterator (such as form widgets) expects uniform structure, and this bug would cause the first two elements to be strings instead of a single tuple.

## Fix

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -whatever_line_number,6 +whatever_line_number,6 @@
     def __iter__(self):
         choices, other = tee(self.choices)
         if not any(value in ("", None) for value, _ in flatten_choices(other)):
-            yield from self.blank_choice
+            yield self.blank_choice
         yield from choices
```

The fix changes `yield from self.blank_choice` to `yield self.blank_choice`, which yields the tuple as a single element instead of unpacking it.