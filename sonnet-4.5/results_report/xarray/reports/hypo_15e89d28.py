from hypothesis import given, strategies as st, settings
from xarray.core.formatting import maybe_truncate, pretty_print

@given(st.text(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=100)
def test_maybe_truncate_at_most_maxlen(text, maxlen):
    result = maybe_truncate(text, maxlen)
    assert len(result) <= maxlen, f"maybe_truncate({text!r}, maxlen={maxlen}) returned {result!r} with length {len(result)}, expected <= {maxlen}"

@given(st.text(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=100)
def test_pretty_print_produces_exact_length(text, numchars):
    result = pretty_print(text, numchars)
    assert len(result) == numchars, f"pretty_print({text!r}, numchars={numchars}) returned {result!r} with length {len(result)}, expected exactly {numchars}"

if __name__ == "__main__":
    print("Running property-based tests with Hypothesis...")
    print("=" * 60)

    print("\nTesting maybe_truncate()...")
    try:
        test_maybe_truncate_at_most_maxlen()
        print("✓ All tests passed for maybe_truncate()")
    except AssertionError as e:
        print(f"✗ Test failed for maybe_truncate()")
        print(f"  {e}")

    print("\nTesting pretty_print()...")
    try:
        test_pretty_print_produces_exact_length()
        print("✓ All tests passed for pretty_print()")
    except AssertionError as e:
        print(f"✗ Test failed for pretty_print()")
        print(f"  {e}")