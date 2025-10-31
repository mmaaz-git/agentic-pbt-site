#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from fastapi.utils import is_body_allowed_for_status_code

@given(st.one_of(st.none(), st.integers(), st.text()))
@settings(max_examples=100)
def test_is_body_allowed_no_crash(status_code):
    """Test that the function doesn't crash with various inputs"""
    try:
        result = is_body_allowed_for_status_code(status_code)
        assert isinstance(result, bool)
        print(f"✓ {status_code!r} -> {result}")
    except Exception as e:
        print(f"✗ {status_code!r} raised {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_is_body_allowed_no_crash()
        print("\nTest passed!")
    except Exception as e:
        print(f"\nTest failed with: {e}")
        import traceback
        traceback.print_exc()