# Bug Report: Flask TaggedJSONSerializer Datetime Microseconds Loss

**Target**: `flask.sessions.TaggedJSONSerializer` (specifically `flask.json.tag.TagDateTime`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TaggedJSONSerializer loses microsecond precision when serializing datetime objects, violating the documented promise of "lossless serialization" and breaking the fundamental round-trip property of serializers.

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

if __name__ == "__main__":
    test_tagged_json_datetime_roundtrip()
```

<details>

<summary>
**Failing input**: `datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 16, in <module>
    test_tagged_json_datetime_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_tagged_json_datetime_roundtrip
    def test_tagged_json_datetime_roundtrip(dt):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 13, in test_tagged_json_datetime_roundtrip
    assert deserialized == dt
           ^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_tagged_json_datetime_roundtrip(
    dt=datetime.datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=datetime.timezone.utc),
)
```
</details>

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

assert deserialized == dt_with_microseconds, f"Expected {dt_with_microseconds}, got {deserialized}"
```

<details>

<summary>
AssertionError: Datetime loses microseconds after round-trip
</summary>
```
Original: 2000-01-01 00:00:00.123456+00:00
Microseconds: 123456
Serialized: {" d":"Sat, 01 Jan 2000 00:00:00 GMT"}
Deserialized: 2000-01-01 00:00:00+00:00
Microseconds after round-trip: 0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 18, in <module>
    assert deserialized == dt_with_microseconds, f"Expected {dt_with_microseconds}, got {deserialized}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 2000-01-01 00:00:00.123456+00:00, got 2000-01-01 00:00:00+00:00
```
</details>

## Why This Is A Bug

This violates Flask's documented behavior in multiple ways:

1. **Breaks the lossless serialization promise**: The module docstring at `/flask/json/tag.py:5` explicitly states "A compact representation for lossless serialization of non-standard JSON types", but microsecond data is lost.

2. **Violates serializer round-trip property**: A fundamental property of serializers is that `loads(dumps(x)) == x` should hold. This fails for any datetime with non-zero microseconds.

3. **Silent data loss**: The serializer silently truncates microseconds to zero without any warning, error, or documentation of this limitation. Users have no indication their data is being corrupted.

4. **Documentation lists datetime as supported**: The TaggedJSONSerializer docstring at line 224-231 lists `datetime.datetime` as a supported type without mentioning any precision limitations.

## Relevant Context

The root cause is in the `TagDateTime` class implementation at `/flask/json/tag.py:205-216`:

- Line 213: `to_json()` uses Werkzeug's `http_date()` function which formats dates in RFC 2822 format
- Line 216: `to_python()` uses Werkzeug's `parse_date()` to parse the RFC 2822 formatted string
- RFC 2822 format only supports second-level precision: "Sat, 01 Jan 2000 00:00:00 GMT"
- Microseconds are truncated during `http_date()` conversion and cannot be recovered

This affects any Flask application that:
- Stores datetime objects in session cookies
- Requires sub-second timestamp precision
- Uses TaggedJSONSerializer for datetime serialization

The Werkzeug functions themselves work correctly according to RFC 2822 specification. The issue is Flask's choice to use an HTTP date format for general datetime serialization where users expect full precision preservation.

## Proposed Fix

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -210,10 +210,10 @@ class TagDateTime(JSONTag):
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        return value.isoformat()

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        return datetime.fromisoformat(value)
```