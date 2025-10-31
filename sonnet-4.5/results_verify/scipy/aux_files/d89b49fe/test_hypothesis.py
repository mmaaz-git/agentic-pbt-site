from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_grayscale_preserves_white(r, g, b):
    """Property: Pure white should convert to pure white in grayscale."""
    weights = [0.21, 0.71, 0.07]
    gray_value = int(weights[0] * r + weights[1] * g + weights[2] * b)

    if r == 255 and g == 255 and b == 255:
        assert gray_value == 255, f"White (255,255,255) converted to {gray_value}, not 255"

# Run the test
if __name__ == "__main__":
    test_grayscale_preserves_white()
    print("Testing specific case: r=255, g=255, b=255")
    test_grayscale_preserves_white(255, 255, 255)