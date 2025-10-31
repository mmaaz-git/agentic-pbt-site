from datetime import datetime, timezone
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

# Test with UTC-aware datetime
utc_dt = datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc)
result_utc = serializer.loads(serializer.dumps(utc_dt))

print("UTC-aware datetime test:")
print(f"Input:  {utc_dt!r}")
print(f"Output: {result_utc!r}")
print(f"Are they equal? {result_utc == utc_dt}")
print()

# Test with naive datetime
naive_dt = datetime(2000, 1, 1, 0, 0)
result_naive = serializer.loads(serializer.dumps(naive_dt))

print("Naive datetime test:")
print(f"Input:  {naive_dt!r}")
print(f"Output: {result_naive!r}")
print(f"Are they equal? {result_naive == naive_dt}")
print()

# Test intermediate serialized form
import json
naive_dumped = serializer.dumps(naive_dt)
utc_dumped = serializer.dumps(utc_dt)

print("Serialized forms:")
print(f"Naive: {json.dumps(naive_dumped, indent=2)}")
print(f"UTC:   {json.dumps(utc_dumped, indent=2)}")