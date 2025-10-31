import pandas.api.types as types
import re
from hypothesis import given, strategies as st, example
import traceback

@given(st.text(min_size=1))
@example('(')  # The specific failing case from the bug report
@example(')')
@example('[')
def test_is_re_compilable_for_valid_patterns(pattern):
    """Test from the bug report"""
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False

    try:
        result = types.is_re_compilable(pattern)
        assert result == can_compile, f"Expected {can_compile} for pattern '{pattern}', got {result}"
        print(f"✓ Pattern '{pattern[:20]}...' -> {result}")
    except Exception as e:
        print(f"✗ Pattern '{pattern[:20]}...' raised {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the test
    test_is_re_compilable_for_valid_patterns()
    print("\nTest completed successfully for all random patterns!")