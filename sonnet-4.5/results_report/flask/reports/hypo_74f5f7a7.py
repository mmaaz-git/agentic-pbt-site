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