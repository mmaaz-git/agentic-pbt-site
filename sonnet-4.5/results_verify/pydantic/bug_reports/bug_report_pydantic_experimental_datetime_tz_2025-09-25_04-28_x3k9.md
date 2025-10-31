# Bug Report: Pydantic Experimental Pipeline datetime_tz() Not Implemented

**Target**: `pydantic.experimental.pipeline._Pipeline.datetime_tz`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `datetime_tz()` method is exposed in the public API but raises `NotImplementedError` when called with an actual timezone, violating the API contract.

## Property-Based Test

```python
import datetime
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

@given(st.datetimes(timezones=st.timezones()))
def test_datetime_tz_should_work(dt):
    pipeline = validate_as(datetime.datetime).datetime_tz(dt.tzinfo)

    class TestModel(BaseModel):
        field: Annotated[datetime.datetime, pipeline]

    model = TestModel(field=dt)
    assert model.field.tzinfo == dt.tzinfo
```

**Failing input**: `Any datetime with a timezone`

## Reproducing the Bug

```python
import datetime
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(datetime.datetime).datetime_tz(datetime.timezone.utc)

class TestModel(BaseModel):
    field: Annotated[datetime.datetime, pipeline]

dt = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
model = TestModel(field=dt)
```

Output:
```
NotImplementedError: Constraining to a specific timezone is not yet supported
```

## Why This Is A Bug

The `datetime_tz()` method is part of the public API (line 290-293 in pipeline.py):

```python
def datetime_tz(
    self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
) -> _Pipeline[_InT, datetime.datetime]:
    return self.constrain(annotated_types.Timezone(tz))
```

However, when this method is called, it creates an `annotated_types.Timezone` constraint with a specific timezone value. The constraint handler at lines 574-575 raises `NotImplementedError` for this case:

```python
else:
    raise NotImplementedError('Constraining to a specific timezone is not yet supported')
```

This is a **contract violation**: the method is publicly exposed and has no indication it doesn't work, but it always fails.

The `# type: ignore` comment on line 293 suggests the developers were aware of a type issue, but the real problem is that the functionality is not implemented.

**Similar methods that DO work:**
- `datetime_tz_naive()` - works correctly
- `datetime_tz_aware()` - works correctly
- `datetime_with_tz()` - works correctly (uses transform instead of constraint)

## Fix

**Option 1**: Remove the method from the public API until it's implemented:

```diff
- def datetime_tz(
-     self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
- ) -> _Pipeline[_InT, datetime.datetime]:
-     return self.constrain(annotated_types.Timezone(tz))
```

**Option 2**: Implement the constraint handler for specific timezones:

```diff
 elif isinstance(constraint, annotated_types.Timezone):
     tz = constraint.tz

     if tz is ...:
         if s and s['type'] == 'datetime':
             s = s.copy()
             s['tz_constraint'] = 'aware'
         else:
             def check_tz_aware(v: object) -> bool:
                 assert isinstance(v, datetime.datetime)
                 return v.tzinfo is not None
             s = _check_func(check_tz_aware, 'timezone aware', s)
     elif tz is None:
         if s and s['type'] == 'datetime':
             s = s.copy()
             s['tz_constraint'] = 'naive'
         else:
             def check_tz_naive(v: object) -> bool:
                 assert isinstance(v, datetime.datetime)
                 return v.tzinfo is None
             s = _check_func(check_tz_naive, 'timezone naive', s)
     else:
-        raise NotImplementedError('Constraining to a specific timezone is not yet supported')
+        def check_specific_tz(v: object) -> bool:
+            assert isinstance(v, datetime.datetime)
+            return v.tzinfo == tz
+        s = _check_func(check_specific_tz, f'timezone {tz}', s)
```

**Option 3**: Mark the method as not implemented in the signature:

```diff
 def datetime_tz(
     self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
 ) -> _Pipeline[_InT, datetime.datetime]:
+    raise NotImplementedError('Constraining to a specific timezone is not yet supported')
-    return self.constrain(annotated_types.Timezone(tz))
```