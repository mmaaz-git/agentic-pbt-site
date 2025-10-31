from hypothesis import given, strategies as st, settings, Verbosity
import pandas as pd
from io import StringIO
import numpy as np

@given(
    st.lists(
        st.tuples(st.integers(), st.integers()),
        min_size=1,
        max_size=50
    )
)
@settings(verbosity=Verbosity.verbose, max_examples=100)
def test_dtype_specification_preserved(rows):
    """Test that dtype specification is correctly preserved without data corruption."""
    csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)
    result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})

    # Check that dtypes are preserved
    assert result['a'].dtype == np.int64
    assert result['b'].dtype == np.int32

    # Check that values are within expected ranges for int32
    for idx, (a_orig, b_orig) in enumerate(rows):
        b_parsed = result['b'].iloc[idx]

        # For int32, values should either:
        # 1. Be correctly parsed if within range
        # 2. Raise an error if out of range
        # They should NOT silently wraparound

        if -2147483648 <= b_orig <= 2147483647:
            # Value is within int32 range, should be preserved
            assert b_parsed == b_orig, f"Value {b_orig} within int32 range but got {b_parsed}"
        else:
            # Value is outside int32 range
            # This SHOULD raise an error, but due to the bug it wraps around
            # We'll check if wraparound occurred
            if b_parsed != b_orig:
                print(f"\n⚠️ BUG DETECTED: Integer overflow!")
                print(f"  Original value: {b_orig}")
                print(f"  Parsed value:   {b_parsed}")
                print(f"  Expected: OverflowError or ValueError")
                print(f"  Actual: Silent wraparound occurred")
                raise AssertionError(
                    f"Integer overflow: {b_orig} silently wrapped to {b_parsed} "
                    f"instead of raising an error"
                )

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test for pandas.read_csv integer overflow...")
    print("=" * 60)
    try:
        test_dtype_specification_preserved()
        print("\n✓ All tests passed (no overflow detected in random samples)")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("\nThis demonstrates the bug where values outside int32 range")
        print("silently wraparound instead of raising an error.")