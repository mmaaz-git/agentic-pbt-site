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