# Bug Report: Flask TaggedJSONSerializer Naive Datetime Round-Trip Failure

**Target**: `flask.json.tag.TaggedJSONSerializer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

TaggedJSONSerializer fails to preserve timezone-awareness for naive datetime objects during serialization round-trips. All naive datetimes are incorrectly converted to UTC-aware datetimes upon deserialization.

## Property-Based Test

```python
from datetime import datetime
from hypothesis import given, strategies as st
from flask.json.tag import TaggedJSONSerializer


@given(st.datetimes())
def test_taggedjson_datetime_roundtrip(data):
    serializer = TaggedJSONSerializer()
    result = serializer.loads(serializer.dumps(data))
    assert result == data, f"Round-trip failed: {result!r} != {data!r}"


if __name__ == "__main__":
    test_taggedjson_datetime_roundtrip()
```

<details>

<summary>
**Failing input**: `datetime.datetime(2000, 1, 1, 0, 0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 14, in <module>
    test_taggedjson_datetime_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 7, in test_taggedjson_datetime_roundtrip
    def test_taggedjson_datetime_roundtrip(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 10, in test_taggedjson_datetime_roundtrip
    assert result == data, f"Round-trip failed: {result!r} != {data!r}"
           ^^^^^^^^^^^^^^
AssertionError: Round-trip failed: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc) != datetime.datetime(2000, 1, 1, 0, 0)
Falsifying example: test_taggedjson_datetime_roundtrip(
    data=datetime.datetime(2000, 1, 1, 0, 0),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

# Test with a naive datetime
naive_dt = datetime(2000, 1, 1, 0, 0)
print(f"Original naive datetime: {naive_dt!r}")
print(f"  tzinfo: {naive_dt.tzinfo}")

# Serialize and deserialize
serialized = serializer.dumps(naive_dt)
print(f"\nSerialized: {serialized}")

result = serializer.loads(serialized)
print(f"\nDeserialized datetime: {result!r}")
print(f"  tzinfo: {result.tzinfo}")

# Test equality
print(f"\nAre they equal? {result == naive_dt}")
print(f"Original is naive: {naive_dt.tzinfo is None}")
print(f"Result is naive: {result.tzinfo is None}")

# This assertion will fail
assert result == naive_dt, f"Round-trip failed: {result!r} != {naive_dt!r}"
```

<details>

<summary>
AssertionError: Round-trip failed - naive datetime becomes UTC-aware
</summary>
```
Original naive datetime: datetime.datetime(2000, 1, 1, 0, 0)
  tzinfo: None

Serialized: {" d":"Sat, 01 Jan 2000 00:00:00 GMT"}

Deserialized datetime: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
  tzinfo: UTC

Are they equal? False
Original is naive: True
Result is naive: False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/repo.py", line 25, in <module>
    assert result == naive_dt, f"Round-trip failed: {result!r} != {naive_dt!r}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc) != datetime.datetime(2000, 1, 1, 0, 0)
```
</details>

## Why This Is A Bug

The TaggedJSONSerializer violates a fundamental contract of serialization: `loads(dumps(x)) == x`. This round-trip property is essential for data preservation during serialization and is a reasonable expectation even if not explicitly documented.

Python's datetime module explicitly distinguishes between naive datetimes (without timezone information) and aware datetimes (with timezone information). This distinction is semantically important:

1. **Naive datetimes** often represent local time, abstract timestamps, or times where timezone is irrelevant
2. **Aware datetimes** represent specific moments in time with explicit timezone context
3. Python's datetime equality operator returns `False` when comparing naive and aware datetimes, even if they represent the same wall clock time

The bug occurs because `TagDateTime.to_python()` (line 216 in flask/json/tag.py) uses `werkzeug.http.parse_date()`, which according to its documentation "always returns a timezone-aware datetime object. If the string doesn't have timezone information, it is assumed to be UTC." This behavior was introduced in Werkzeug 2.0.

Applications that rely on naive datetimes will experience silent data corruption when these values pass through Flask's session serialization or any other use of TaggedJSONSerializer. For example, an application storing local times without timezone context will suddenly find all its times converted to UTC-aware datetimes after deserialization.

## Relevant Context

The TaggedJSONSerializer is a critical component in Flask, used by the SecureCookieSessionInterface to serialize session data. The serializer supports various non-standard JSON types including datetime objects, making this bug likely to affect many Flask applications.

The serialization format uses HTTP date strings (RFC 2822 format) which inherently lose the naive/aware distinction since they always include "GMT" in the string representation. This makes the bug more subtle since the serialized format looks correct but loses critical type information.

Flask source code location: `/flask/json/tag.py`, specifically:
- Line 205-217: `TagDateTime` class implementation
- Line 213: `to_json()` uses `http_date()` to serialize
- Line 216: `to_python()` uses `parse_date()` which always returns aware datetimes

Related Werkzeug documentation notes that `parse_date` was changed in version 2.0 to always return timezone-aware objects, which likely introduced or exacerbated this issue.

## Proposed Fix

The fix requires preserving the naive/aware distinction in the serialized format. Here's a backward-compatible approach that adds metadata to track whether the original datetime was naive:

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -210,10 +210,20 @@ class TagDateTime(JSONTag):
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        # Preserve naive/aware distinction by adding metadata
+        is_naive = value.tzinfo is None
+        return {"dt": http_date(value), "naive": is_naive}

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        # Handle both old string format and new dict format for compatibility
+        if isinstance(value, dict):
+            dt = parse_date(value["dt"])
+            # If original was naive, strip timezone
+            if value.get("naive", False) and dt is not None and dt.tzinfo is not None:
+                return dt.replace(tzinfo=None)
+            return dt
+        # Backward compatibility: old format always becomes aware
+        return parse_date(value)
```

This fix maintains backward compatibility by checking if the deserialized value is a dict (new format) or string (old format), ensuring existing serialized data continues to work while fixing the issue for new serializations.