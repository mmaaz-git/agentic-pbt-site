#!/usr/bin/env python3
"""Property-based test demonstrating ujson round-trip bug."""

from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads
import math


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_ujson_roundtrip_preserves_finiteness(value):
    serialized = ujson_dumps(value)
    deserialized = ujson_loads(serialized)

    if math.isfinite(value):
        assert math.isfinite(deserialized), \
            f"Round-trip should preserve finiteness: {value} -> {serialized} -> {deserialized}"


if __name__ == "__main__":
    # Run the test
    test_ujson_roundtrip_preserves_finiteness()