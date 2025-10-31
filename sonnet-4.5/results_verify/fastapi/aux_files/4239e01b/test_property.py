from hypothesis import given, strategies as st, settings
from attrs import converters

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_to_bool_rejects_floats(val):
    try:
        result = converters.to_bool(val)
        print(f"to_bool({val!r}) returned {result} but should have raised ValueError")
        assert False, f"to_bool({val!r}) should raise ValueError but returned {result}"
    except ValueError:
        pass  # This is expected

if __name__ == "__main__":
    test_to_bool_rejects_floats()