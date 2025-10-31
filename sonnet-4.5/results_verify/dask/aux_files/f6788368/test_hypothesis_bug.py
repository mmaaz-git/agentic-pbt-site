from hypothesis import given, strategies as st

@given(
    st.integers(min_value=1, max_value=10**9),
    st.integers(min_value=1, max_value=10**6)
)
def test_blocksize_calculation_terminates(size, blocksize):
    if size <= blocksize:
        return

    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []
    iterations = 0

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
        iterations += 1
        assert iterations < 10000, f"Infinite loop detected for size={size}, blocksize={blocksize}"

    length.append(size - off[-1])

# Test the specific failing case mentioned
def test_specific_failing_case():
    size = 1000000000
    blocksize = 333333

    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []
    iterations = 0

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
        iterations += 1
        assert iterations < 10000, f"Infinite loop detected for size={size}, blocksize={blocksize}"

    length.append(size - off[-1])

if __name__ == "__main__":
    # Run the specific test
    try:
        test_specific_failing_case()
        print("Test passed - no infinite loop detected")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run hypothesis test with the specific failing input
    test_blocksize_calculation_terminates(1000000000, 333333)