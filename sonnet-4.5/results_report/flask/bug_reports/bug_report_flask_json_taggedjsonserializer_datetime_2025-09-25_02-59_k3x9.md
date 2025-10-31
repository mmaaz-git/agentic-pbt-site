# Bug Report: Flask TaggedJSONSerializer Naive Datetime Round-Trip Failure

**Target**: `flask.json.tag.TaggedJSONSerializer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

TaggedJSONSerializer fails to preserve timezone-awareness for naive datetime objects during serialization round-trips. Naive datetimes are incorrectly converted to UTC-aware datetimes.

## Property-Based Test

```python
from datetime import datetime
from hypothesis import given, strategies as st
from flask.json.tag import TaggedJSONSerializer


@given(st.datetimes())
def test_taggedjson_datetime_roundtrip(data):
    serializer = TaggedJSONSerializer()
    result = serializer.loads(serializer.dumps(data))
    assert result == data
```

**Failing input**: `datetime.datetime(2000, 1, 1, 0, 0)` (any naive datetime)

## Reproducing the Bug

```python
from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

naive_dt = datetime(2000, 1, 1, 0, 0)
result = serializer.loads(serializer.dumps(naive_dt))

print(f"Input:  {naive_dt!r}")
print(f"Output: {result!r}")

assert result == naive_dt
```

**Output:**
```
Input:  datetime.datetime(2000, 1, 1, 0, 0)
Output: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
AssertionError: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc) != datetime.datetime(2000, 1, 1, 0, 0)
```

## Why This Is A Bug

The TaggedJSONSerializer violates the fundamental round-trip property: `loads(dumps(x)) == x`. Python distinguishes between naive datetimes (no timezone) and aware datetimes (with timezone), and this distinction has semantic meaning. Applications that use naive datetimes to represent local time or abstract time values will experience data corruption when these datetimes are unexpectedly converted to UTC-aware datetimes.

The root cause is in `TagDateTime.to_python()` which uses `werkzeug.http.parse_date()`. This function explicitly converts all naive datetimes to UTC-aware ones (see werkzeug/http.py:1007-1008).

## Fix

The TagDateTime implementation needs to preserve the naive/aware distinction. One approach is to add a marker in the serialized format to indicate whether the original datetime was naive:

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -210,10 +210,16 @@ class TagDateTime(JSONTag):
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        is_naive = value.tzinfo is None
+        return {"date": http_date(value), "naive": is_naive}

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        if isinstance(value, dict):
+            dt = parse_date(value["date"])
+            if value.get("naive", False) and dt.tzinfo is not None:
+                return dt.replace(tzinfo=None)
+            return dt
+        return parse_date(value)
```

This fix maintains backward compatibility by checking if the value is a dict, while still handling old serialized formats.