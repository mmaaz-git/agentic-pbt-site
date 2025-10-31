# Bug Report: Flask TaggedJSONSerializer Datetime Microseconds Loss

**Target**: `flask.sessions.TaggedJSONSerializer` (specifically `flask.json.tag.TagDateTime`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TaggedJSONSerializer silently loses microsecond precision when serializing and deserializing datetime objects, violating the round-trip property that serializers should maintain.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask.sessions import TaggedJSONSerializer
from datetime import timezone

@given(st.datetimes(timezones=st.just(timezone.utc)))
def test_tagged_json_datetime_roundtrip(dt):
    """Datetimes should round-trip perfectly"""
    serializer = TaggedJSONSerializer()

    serialized = serializer.dumps(dt)
    deserialized = serializer.loads(serialized)

    assert deserialized == dt
```

**Failing input**: `datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc)` (any datetime with non-zero microseconds)

## Reproducing the Bug

```python
from flask.sessions import TaggedJSONSerializer
from datetime import datetime, timezone

serializer = TaggedJSONSerializer()

dt_with_microseconds = datetime(2000, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)

print(f"Original: {dt_with_microseconds}")
print(f"Microseconds: {dt_with_microseconds.microsecond}")

serialized = serializer.dumps(dt_with_microseconds)
print(f"Serialized: {serialized}")

deserialized = serializer.loads(serialized)
print(f"Deserialized: {deserialized}")
print(f"Microseconds after round-trip: {deserialized.microsecond}")

assert deserialized == dt_with_microseconds
```

Output:
```
Original: 2000-01-01 00:00:00.123456+00:00
Microseconds: 123456
Serialized: {" d":"Sat, 01 Jan 2000 00:00:00 GMT"}
Deserialized: 2000-01-01 00:00:00+00:00
Microseconds after round-trip: 0
AssertionError
```

## Why This Is A Bug

1. **Violates round-trip property**: Serializers should preserve data through serialize/deserialize cycles. `loads(dumps(x)) == x` should hold, but it doesn't for datetimes with microseconds.

2. **Silent data loss**: The microseconds are silently truncated without warning or error, which could lead to subtle bugs in applications.

3. **Undocumented limitation**: The TaggedJSONSerializer docstring claims to support datetime objects but doesn't mention precision loss.

4. **Real-world impact**: Applications storing timestamps in Flask sessions will lose sub-second precision, which could affect:
   - High-frequency event logging
   - Rate limiting with sub-second granularity
   - Any application requiring precise timestamps

## Fix

The root cause is in `flask/json/tag.py` where `TagDateTime` uses `http_date()` and `parse_date()` from Werkzeug, which use RFC 2822 format (second precision only). The fix is to use ISO 8601 format instead, which preserves microseconds.

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -231,10 +231,10 @@ class TagDateTime(JSONTag):
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        return value.isoformat()

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        return datetime.fromisoformat(value)
```

This change:
- Preserves microsecond precision
- Maintains timezone information
- Uses standard Python datetime methods (available since Python 3.7)
- Produces more compact JSON (e.g., `"2000-01-01T00:00:00.123456+00:00"` vs `"Sat, 01 Jan 2000 00:00:00 GMT"`)