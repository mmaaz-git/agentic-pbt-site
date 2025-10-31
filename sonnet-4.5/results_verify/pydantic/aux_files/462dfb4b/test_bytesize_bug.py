from hypothesis import given, strategies as st
from pydantic.v1.types import ByteSize


@given(scalar=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_bytesize_precision_loss(scalar):
    bs = ByteSize.validate(f"{scalar}b")
    result = bs.to('b')
    assert result == scalar, f"ByteSize lost precision: {scalar}b -> {bs} -> {result}b"

# Run the test
if __name__ == "__main__":
    test_bytesize_precision_loss()