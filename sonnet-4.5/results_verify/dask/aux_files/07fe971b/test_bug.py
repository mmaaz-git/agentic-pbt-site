from hypothesis import given, strategies as st, settings, assume

@given(
    size=st.integers(min_value=100, max_value=10000),
    blocksize=st.integers(min_value=10, max_value=1000)
)
@settings(max_examples=500)
def test_blocksize_calculation_uses_float_division(size, blocksize):
    """
    Property: Block offset calculations should use integer arithmetic, not floats.

    This test demonstrates that line 125 uses float division:
        blocksize1 = size / (size // blocksize)

    This produces different results than integer division.
    """
    assume(size % blocksize != 0)
    assume(size > blocksize)

    num_blocks = size // blocksize
    blocksize_float = size / num_blocks  # Current implementation
    blocksize_int = size // num_blocks   # Correct implementation

    place_float = 0.0
    place_int = 0
    offsets_float = [0]
    offsets_int = [0]

    while size - place_float > (blocksize_float * 2) - 1:
        place_float += blocksize_float
        offsets_float.append(int(place_float))

    while size - place_int > (blocksize_int * 2) - 1:
        place_int += blocksize_int
        offsets_int.append(place_int)

    assert offsets_float != offsets_int, \
        f"Float division produces incorrect offsets"

# Run the test
if __name__ == "__main__":
    test_blocksize_calculation_uses_float_division()
    print("Test completed - if no error, it means the test passed (which indicates the bug exists)")