#!/usr/bin/env python3
from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

dt_naive = datetime(2000, 1, 1, 0, 0)
print(f"Original:     {repr(dt_naive)}")

serialized = serializer.dumps(dt_naive)
print(f"Serialized:   {repr(serialized)}")

deserialized = serializer.loads(serialized)

print(f"Deserialized: {repr(deserialized)}")
print(f"Equal:        {dt_naive == deserialized}")

# Check the timezone info
print(f"Original tzinfo:     {dt_naive.tzinfo}")
print(f"Deserialized tzinfo: {deserialized.tzinfo}")