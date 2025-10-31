# Bug Report: flask.json.tag.TaggedJSONSerializer Naive Datetime Corruption

**Target**: `flask.json.tag.TaggedJSONSerializer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Flask's TaggedJSONSerializer violates its documented "lossless serialization" contract by silently converting naive datetime objects into timezone-aware datetimes with UTC timezone during roundtrip serialization.

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

if __name__ == "__main__":
    test_datetime_roundtrip()
```

<details>

<summary>
**Failing input**: `datetime.datetime(2000, 1, 1, 0, 0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 15, in <module>
    test_datetime_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 7, in test_datetime_roundtrip
    def test_datetime_roundtrip(dt):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 12, in test_datetime_roundtrip
    assert deserialized == dt
           ^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_datetime_roundtrip(
    dt=datetime.datetime(2000, 1, 1, 0, 0),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

# Test with a naive datetime (no timezone info)
dt_naive = datetime(2000, 1, 1, 0, 0)
print(f"Original:     {repr(dt_naive)}")
print(f"  tzinfo:     {dt_naive.tzinfo}")

# Serialize the datetime
serialized = serializer.dumps(dt_naive)
print(f"\nSerialized:   {serialized}")

# Deserialize it back
deserialized = serializer.loads(serialized)
print(f"\nDeserialized: {repr(deserialized)}")
print(f"  tzinfo:     {deserialized.tzinfo}")

# Check if they're equal
print(f"\nRoundtrip equality check:")
print(f"  dt_naive == deserialized: {dt_naive == deserialized}")

# Show the difference
if dt_naive != deserialized:
    print(f"\nDifference:")
    print(f"  Original is naive:     {dt_naive.tzinfo is None}")
    print(f"  Deserialized is aware: {deserialized.tzinfo is not None}")
    if deserialized.tzinfo is not None:
        print(f"  Deserialized timezone: {deserialized.tzinfo}")
```

<details>

<summary>
Roundtrip serialization fails - naive datetime becomes timezone-aware
</summary>
```
Original:     datetime.datetime(2000, 1, 1, 0, 0)
  tzinfo:     None

Serialized:   {" d":"Sat, 01 Jan 2000 00:00:00 GMT"}

Deserialized: datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
  tzinfo:     UTC

Roundtrip equality check:
  dt_naive == deserialized: False

Difference:
  Original is naive:     True
  Deserialized is aware: True
  Deserialized timezone: UTC
```
</details>

## Why This Is A Bug

This violates the explicit "lossless serialization" contract documented at the top of `/flask/json/tag.py` (line 5). The TaggedJSONSerializer claims to provide "A compact representation for lossless serialization of non-standard JSON types," yet it fundamentally changes the nature of naive datetime objects.

The root cause is in the `TagDateTime` class implementation (lines 205-217 of `tag.py`):
- `to_json()` uses Werkzeug's `http_date()` function which assumes naive datetimes are UTC and formats them as RFC 2822 date strings with "GMT" suffix
- `to_python()` uses Werkzeug's `parse_date()` function which, as of Werkzeug 2.0, always returns timezone-aware datetime objects with UTC timezone

This is particularly problematic because:
1. **Semantic corruption**: Naive and timezone-aware datetimes have different meanings. A naive datetime represents a "local" or "abstract" time without timezone context, while a timezone-aware datetime represents a specific moment in time.
2. **Session data integrity**: Flask uses TaggedJSONSerializer for session serialization, meaning any naive datetimes stored in sessions will be corrupted.
3. **Silent failure**: The transformation happens silently without warning, making bugs difficult to track down.
4. **Equality breaks**: Applications comparing original and deserialized datetimes will unexpectedly fail.

## Relevant Context

- **Werkzeug version**: 3.1.3 (behavior changed in 2.0)
- **Impact**: Affects all Flask applications using TaggedJSONSerializer with naive datetimes
- **Documentation location**: `/flask/json/tag.py` lines 5, 220-232
- **Class affected**: `TagDateTime` (lines 205-217)
- **Related GitHub issue**: pallets/flask#4466 indicates this is a known issue

The Werkzeug documentation for `parse_date()` explicitly states: "always returns a timezone-aware datetime object. If the string doesn't have timezone information, it is assumed to be UTC." This makes the current implementation incompatible with lossless serialization of naive datetimes.

## Proposed Fix

The fix requires preserving the timezone awareness state during serialization and restoring it during deserialization:

```diff
--- a/flask/json/tag.py
+++ b/flask/json/tag.py
@@ -209,10 +209,16 @@ class TagDateTime(JSONTag):
     def check(self, value: t.Any) -> bool:
         return isinstance(value, datetime)

     def to_json(self, value: t.Any) -> t.Any:
-        return http_date(value)
+        # Store both the HTTP date and whether the original was timezone-aware
+        return {
+            "dt": http_date(value),
+            "tz": value.tzinfo is not None
+        }

     def to_python(self, value: t.Any) -> t.Any:
-        return parse_date(value)
+        dt = parse_date(value["dt"])
+        # If original was naive, remove the UTC timezone added by parse_date
+        if not value["tz"] and dt is not None and dt.tzinfo is not None:
+            dt = dt.replace(tzinfo=None)
+        return dt
```