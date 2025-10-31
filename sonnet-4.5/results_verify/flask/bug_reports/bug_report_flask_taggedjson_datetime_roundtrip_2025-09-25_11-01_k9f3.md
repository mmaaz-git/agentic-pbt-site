# Bug Report: Flask TaggedJSONSerializer Naive Datetime Corruption

**Target**: `flask.json.tag.TaggedJSONSerializer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TaggedJSONSerializer violates its "lossless serialization" contract by adding UTC timezone information to naive datetime objects during deserialization, corrupting the original data.

## Property-Based Test

```python
from datetime import datetime
from hypothesis import given, strategies as st
from flask.json.tag import TaggedJSONSerializer


@given(st.datetimes())
def test_datetime_roundtrip(dt):
    serializer = TaggedJSONSerializer()
    serialized = serializer.dumps(dt)
    deserialized = serializer.loads(serialized)

    assert deserialized == dt
```

**Failing input**: `datetime.datetime(2000, 1, 1, 0, 0)` (any naive datetime)

## Reproducing the Bug

```python
from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

dt_naive = datetime(2000, 1, 1, 0, 0)
print(f"Original:     {repr(dt_naive)}")

serialized = serializer.dumps(dt_naive)
deserialized = serializer.loads(serialized)

print(f"Deserialized: {repr(deserialized)}")
print(f"Equal:        {dt_naive == deserialized}")
```

Output:
```
Original:     datetime.datetime(2000, 1, 1, 0, 0)
Deserialized: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
Equal:        False
```

## Why This Is A Bug

The TaggedJSONSerializer documentation (line 5 of `flask/json/tag.py`) explicitly states it provides "lossless serialization" of non-standard JSON types. However, naive datetime objects (datetimes without timezone info) are deserialized as timezone-aware datetimes with UTC timezone, violating the lossless property.

This occurs because:
1. `TagDateTime.to_json()` uses `http_date()` which formats the datetime as an HTTP date string
2. `TagDateTime.to_python()` uses `parse_date()` which always returns a timezone-aware datetime with UTC

This breaks the round-trip property and corrupts session data containing naive datetimes.

## Fix

The fix requires preserving timezone information. Store whether the original datetime was naive or aware, and restore that property during deserialization.

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -209,10 +209,16 @@ class TagDateTime(JSONTag):
     def check(self, value: t.Any) -> bool:
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        # Store timezone awareness flag with the serialized value
+        return {"dt": http_date(value), "tz": value.tzinfo is not None}

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        dt = parse_date(value["dt"])
+        # If original was naive, remove the UTC timezone added by parse_date
+        if not value["tz"] and dt.tzinfo is not None:
+            dt = dt.replace(tzinfo=None)
+        return dt
```