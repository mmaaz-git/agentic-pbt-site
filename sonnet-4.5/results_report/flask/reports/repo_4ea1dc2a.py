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