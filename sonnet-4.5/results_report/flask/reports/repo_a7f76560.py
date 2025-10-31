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