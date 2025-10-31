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