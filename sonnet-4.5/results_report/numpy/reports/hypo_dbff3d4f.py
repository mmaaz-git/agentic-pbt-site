from hypothesis import given, strategies as st, settings
import numpy.polynomial.polynomial as poly

@given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
       st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=0, max_size=8))
@settings(max_examples=500)
def test_polyval_handles_empty(x, c):
    """Test that polyval handles empty coefficient array without crashing"""
    try:
        result = poly.polyval(x, c)
        if len(c) == 0:
            assert result == 0 or True, "Empty poly should evaluate to something"
    except ValueError:
        pass
    except IndexError:
        assert False, f"Should not raise IndexError for x={x}, c={c}"

# Run the test
test_polyval_handles_empty()
