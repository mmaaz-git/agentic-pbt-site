# Bug Report: django.db.models.functions TruncBase Unreachable Code

**Target**: `django.db.models.functions.datetime.TruncBase.convert_value`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `TruncBase.convert_value` method contains unreachable code at line 358 where it checks `if value is None:` inside an `elif isinstance(value, datetime):` block. Since a datetime instance cannot be None, this condition will always be False and the code block will never execute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from datetime import datetime
from django.db.models.functions.datetime import TruncBase
from django.db.models.fields import DateField
from django.db.models.expressions import Value


@st.composite
def datetime_values(draw):
    year = draw(st.integers(min_value=1900, max_value=2100))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    return datetime(year, month, day, hour, minute, second)


@given(dt=datetime_values())
def test_convert_value_never_reaches_none_check(dt):
    trunc = TruncBase(Value(dt))
    trunc.output_field = DateField()

    result = trunc.convert_value(dt, None, None)

    assert result == dt.date()
```

**Failing input**: N/A - this is dead code, not a functional bug

## Reproducing the Bug

```python
from datetime import datetime

value = datetime(2023, 1, 15, 12, 30, 45)

if isinstance(value, datetime):
    if value is None:
        print("This will NEVER print")
    else:
        print("This will ALWAYS execute if isinstance check passes")
```

## Why This Is A Bug

The code at lines 357-363 in `django/db/models/functions/datetime.py`:

```python
elif isinstance(value, datetime):
    if value is None:  # Line 358 - UNREACHABLE
        pass
    elif isinstance(self.output_field, DateField):
        value = value.date()
    elif isinstance(self.output_field, TimeField):
        value = value.time()
```

The condition `if value is None:` on line 358 is logically impossible because:
1. The outer condition requires `isinstance(value, datetime)` to be True
2. If `value` is an instance of `datetime`, it cannot be `None`
3. Therefore, `value is None` will always evaluate to `False`

This is dead code that should be removed. It likely indicates a logic error during development or leftover code from refactoring.

## Fix

```diff
--- a/django/db/models/functions/datetime.py
+++ b/django/db/models/functions/datetime.py
@@ -355,9 +355,7 @@ class TruncBase(TimezoneMixin, Transform):
                     "zone definitions for your database installed?"
                 )
         elif isinstance(value, datetime):
-            if value is None:
-                pass
-            elif isinstance(self.output_field, DateField):
+            if isinstance(self.output_field, DateField):
                 value = value.date()
             elif isinstance(self.output_field, TimeField):
                 value = value.time()
```