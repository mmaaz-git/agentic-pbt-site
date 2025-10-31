from hypothesis import given, strategies as st, settings, example
from numpy.f2py import symbolic
import traceback


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=500)
@example('(')  # Add the specific failing case
def test_parenthesis_replacement_round_trip(s):
    try:
        new_s, mapping = symbolic.replace_parenthesis(s)
        reconstructed = symbolic.unreplace_parenthesis(new_s, mapping)
        assert s == reconstructed
        return True
    except RecursionError:
        print(f"RecursionError for input: {s!r}")
        return False
    except ValueError as e:
        # ValueError might be expected for unbalanced parentheses
        print(f"ValueError for input {s!r}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for input {s!r}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

# Run the test
print("Running hypothesis test...")
try:
    test_parenthesis_replacement_round_trip()
    print("Test completed")
except Exception as e:
    print(f"Test failed with error: {e}")