from datetime import datetime
from flask.json.tag import TaggedJSONSerializer

serializer = TaggedJSONSerializer()

naive_dt = datetime(2000, 1, 1, 0, 0)
result = serializer.loads(serializer.dumps(naive_dt))

print(f"Input:  {naive_dt!r}")
print(f"Output: {result!r}")
print(f"Input tzinfo: {naive_dt.tzinfo}")
print(f"Output tzinfo: {result.tzinfo}")
print(f"Are they equal? {result == naive_dt}")

try:
    assert result == naive_dt
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {result!r} != {naive_dt!r}")