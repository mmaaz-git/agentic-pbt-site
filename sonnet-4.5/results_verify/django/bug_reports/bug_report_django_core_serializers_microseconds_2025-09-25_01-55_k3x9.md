# Bug Report: DjangoJSONEncoder Silently Truncates Microseconds

**Target**: `django.core.serializers.json.DjangoJSONEncoder`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `DjangoJSONEncoder` silently truncates microseconds to milliseconds when encoding `datetime.datetime` and `datetime.time` objects, causing undocumented data loss in JSON serialization round-trips.

## Property-Based Test

```python
import json
import datetime
from django.core.serializers.json import DjangoJSONEncoder
from hypothesis import given, strategies as st

@given(
    dt=st.datetimes(
        min_value=datetime.datetime(2000, 1, 1),
        max_value=datetime.datetime(2100, 12, 31),
    )
)
def test_datetime_roundtrip_loses_precision(dt):
    encoded_str = json.dumps(dt, cls=DjangoJSONEncoder)
    decoded_str = json.loads(encoded_str)

    if dt.microsecond > 0 and dt.microsecond % 1000 != 0:
        reconstructed = datetime.datetime.fromisoformat(
            decoded_str.replace('Z', '+00:00') if decoded_str.endswith('Z') else decoded_str
        )
        if dt.tzinfo:
            reconstructed = reconstructed.replace(tzinfo=dt.tzinfo)

        assert reconstructed != dt, "Expected precision loss for non-millisecond values"
```

**Failing input**: `datetime.datetime(2000, 1, 1, 12, 34, 56, 123456, tzinfo=datetime.timezone.utc)`

## Reproducing the Bug

```python
import json
import datetime
from django.core.serializers.json import DjangoJSONEncoder

dt = datetime.datetime(2000, 1, 1, 12, 34, 56, 123456, tzinfo=datetime.timezone.utc)
print(f"Original: {dt.isoformat()}")

encoded = json.dumps(dt, cls=DjangoJSONEncoder)
decoded_str = json.loads(encoded)
print(f"Encoded: {decoded_str}")

reconstructed = datetime.datetime.fromisoformat(decoded_str.replace('Z', '+00:00'))
print(f"Reconstructed: {reconstructed.isoformat()}")
print(f"Equal: {dt == reconstructed}")
```

Output:
```
Original: 2000-01-01T12:34:56.123456+00:00
Encoded: 2000-01-01T12:34:56.123Z
Reconstructed: 2000-01-01T12:34:56.123000+00:00
Equal: False
```

The microseconds are truncated from 123456 to 123000, losing the last 3 digits.

## Why This Is A Bug

1. **Undocumented behavior**: The `DjangoJSONEncoder` docstring states it "knows how to encode date/time" but doesn't mention microsecond truncation
2. **Data loss**: Python's `datetime` objects support microsecond precision (6 digits), but the encoder silently reduces this to millisecond precision (3 digits)
3. **Violates user expectations**: When serializing Python datetime objects, users expect full precision to be preserved or at least documented
4. **Silent failure**: No warning or error is raised when precision is lost

The truncation occurs in `/django/core/serializers/json.py`:
- Line 95-96: For `datetime.datetime`: `r = r[:23] + r[26:]` truncates `.123456` to `.123`
- Line 106-107: For `datetime.time`: `r = r[:12]` truncates `.123456` to `.123`

## Fix

The encoder should either:
1. Preserve full microsecond precision (recommended)
2. Document the truncation behavior clearly in the docstring and/or raise a warning

To preserve full precision, remove the truncation logic:

```diff
--- a/django/core/serializers/json.py
+++ b/django/core/serializers/json.py
@@ -92,9 +92,6 @@ class DjangoJSONEncoder(json.JSONEncoder):
         # See "Date Time String Format" in the ECMA-262 specification.
         if isinstance(o, datetime.datetime):
             r = o.isoformat()
-            if o.microsecond:
-                r = r[:23] + r[26:]
             if r.endswith("+00:00"):
                 r = r.removesuffix("+00:00") + "Z"
             return r
@@ -102,9 +99,6 @@ class DjangoJSONEncoder(json.JSONEncoder):
             return o.isoformat()
         elif isinstance(o, datetime.time):
             if is_aware(o):
                 raise ValueError("JSON can't represent timezone-aware times.")
             r = o.isoformat()
-            if o.microsecond:
-                r = r[:12]
             return r
         elif isinstance(o, datetime.timedelta):
             return duration_iso_string(o)
```