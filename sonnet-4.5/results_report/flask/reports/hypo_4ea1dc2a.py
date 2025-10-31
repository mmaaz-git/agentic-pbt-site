from datetime import datetime
from hypothesis import given, strategies as st
from flask.json.tag import TaggedJSONSerializer


@given(st.datetimes())
def test_taggedjson_datetime_roundtrip(data):
    serializer = TaggedJSONSerializer()
    result = serializer.loads(serializer.dumps(data))
    assert result == data, f"Round-trip failed: {result!r} != {data!r}"


if __name__ == "__main__":
    test_taggedjson_datetime_roundtrip()