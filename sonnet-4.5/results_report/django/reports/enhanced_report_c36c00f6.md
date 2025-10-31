# Bug Report: django.db.models.functions.NthValue Grammatical Error in Validation Message

**Target**: `django.db.models.functions.window.NthValue.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The NthValue class in Django's window functions contains a grammatical error in its validation error message, producing "NthValue requires a positive integer as for nth" instead of the grammatically correct "for nth".

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from django.db.models.expressions import F
from django.db.models.functions import NthValue
import pytest


@given(st.integers(max_value=0))
@example(0)  # Ensure we test with 0
@example(-1)  # Ensure we test with -1
def test_nthvalue_error_message_grammar(nth):
    """Test that NthValue raises ValueError with grammatically incorrect message."""
    with pytest.raises(ValueError) as exc_info:
        NthValue(F('field'), nth=nth)

    error_msg = str(exc_info.value)
    assert "positive integer" in error_msg
    # This assertion demonstrates the bug - "as for nth" is grammatically incorrect
    assert "as for nth" in error_msg
```

<details>

<summary>
**Failing input**: `nth=0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/0
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_nthvalue_error_message_grammar PASSED                      [100%]

============================== 1 passed in 0.18s ===============================
```
</details>

## Reproducing the Bug

```python
from django.db.models.expressions import F
from django.db.models.functions import NthValue

# Test with nth=0
try:
    NthValue(F('field'), nth=0)
except ValueError as e:
    print(f"Test with nth=0:")
    print(f"Error message: {str(e)}")
    print()

# Test with nth=-1
try:
    NthValue(F('field'), nth=-1)
except ValueError as e:
    print(f"Test with nth=-1:")
    print(f"Error message: {str(e)}")
    print()

# Test with nth=None
try:
    NthValue(F('field'), nth=None)
except ValueError as e:
    print(f"Test with nth=None:")
    print(f"Error message: {str(e)}")
    print()

# For comparison, check the LagLeadFunction error message
from django.db.models.functions import Lag

try:
    Lag(F('field'), offset=0)
except ValueError as e:
    print(f"Comparison - Lag function with offset=0:")
    print(f"Error message: {str(e)}")
    print()

print("\nAnalysis:")
print("The NthValue error message contains 'as for nth' which is grammatically incorrect.")
print("It should be either 'for nth' or 'as the nth parameter'.")
print("Note how the Lag function correctly uses 'for the offset'.")
```

<details>

<summary>
ValueError raised with grammatically incorrect message
</summary>
```
Test with nth=0:
Error message: NthValue requires a positive integer as for nth.

Test with nth=-1:
Error message: NthValue requires a positive integer as for nth.

Test with nth=None:
Error message: NthValue requires a positive integer as for nth.

Comparison - Lag function with offset=0:
Error message: Lag requires a positive integer for the offset.


Analysis:
The NthValue error message contains 'as for nth' which is grammatically incorrect.
It should be either 'for nth' or 'as the nth parameter'.
Note how the Lag function correctly uses 'for the offset'.
```
</details>

## Why This Is A Bug

This violates expected behavior because the error message contains an objectively incorrect grammatical construction. The phrase "as for nth" combines two prepositions incorrectly, creating a non-standard English phrase. This is inconsistent with Django's typically high standards for user-facing messages and contradicts the pattern used by similar functions in the same module.

Specifically:
1. The LagLeadFunction class in the same file (line 47) uses the grammatically correct phrase "requires a positive integer for the offset"
2. The phrase "as for nth" appears to be an accidental combination of two valid phrasings: "as the nth parameter" and "for nth"
3. Django's documentation and other error messages maintain proper grammar throughout

While the validation logic functions correctly and rejects invalid inputs as expected, the poor grammar in the error message reduces code quality and could confuse users, particularly non-native English speakers who rely on correct grammar patterns.

## Relevant Context

The bug is located in `/django/db/models/functions/window.py` at line 85. The NthValue class is part of Django's window functions, which are used in database queries to perform calculations across a set of table rows that are somehow related to the current row.

Django documentation for NthValue: The function computes the row relative to the offset nth (must be a positive value) within the window. The nth parameter must be a positive integer and defaults to 1.

Code location: https://github.com/django/django/blob/main/django/db/models/functions/window.py#L85

The inconsistency is particularly notable because the same file contains the correct pattern just 38 lines earlier in the LagLeadFunction class, suggesting this was an unintentional typo rather than a deliberate choice.

## Proposed Fix

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