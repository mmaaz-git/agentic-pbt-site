# Bug Report: django.db.models.functions NthValue Error Message Typo

**Target**: `django.db.models.functions.window.NthValue.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NthValue` class has a grammatical error in its error message. When validation fails, it produces the message "NthValue requires a positive integer as for nth" which contains the incorrect phrase "as for nth" instead of the grammatically correct "for nth" or "as nth".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.models.expressions import F
from django.db.models.functions import NthValue
import pytest


@given(st.integers(max_value=0))
def test_nthvalue_error_message_grammar(nth):
    with pytest.raises(ValueError) as exc_info:
        NthValue(F('field'), nth=nth)

    error_msg = str(exc_info.value)
    assert "positive integer" in error_msg
    assert "as for nth" in error_msg
```

**Failing input**: `nth=0` (or any non-positive integer)

## Reproducing the Bug

```python
from django.db.models.expressions import F
from django.db.models.functions import NthValue

try:
    NthValue(F('field'), nth=0)
except ValueError as e:
    print(str(e))
```

Output:
```
NthValue requires a positive integer as for nth.
```

The phrase "as for nth" is grammatically incorrect.

## Why This Is A Bug

The error message at line 84-86 in `django/db/models/functions/window.py`:

```python
raise ValueError(
    "%s requires a positive integer as for nth." % self.__class__.__name__
)
```

Contains the grammatically incorrect phrase "as for nth". This should be one of:
- "for nth" (concise)
- "for the nth parameter" (explicit)
- "as the nth parameter" (alternative phrasing)

The current phrasing appears to be a typo where "as" and "for" were accidentally combined.

## Fix

```diff
--- a/django/db/models/functions/window.py
+++ b/django/db/models/functions/window.py
@@ -82,7 +82,7 @@ class NthValue(Func):
             )
         if nth is None or nth <= 0:
             raise ValueError(
-                "%s requires a positive integer as for nth." % self.__class__.__name__
+                "%s requires a positive integer for nth." % self.__class__.__name__
             )
         super().__init__(expression, nth, **extra)
```