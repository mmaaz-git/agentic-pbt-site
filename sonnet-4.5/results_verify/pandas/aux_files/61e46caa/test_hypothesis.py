#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st, example
from pandas.core.dtypes.common import ensure_python_int
import pytest

@given(st.floats(allow_nan=True, allow_infinity=True))
@example(float('inf'))
@example(float('-inf'))
def test_ensure_python_int_raises_typeerror_for_invalid_floats(value):
    """Test that ensure_python_int raises TypeError for values that can't be converted to int."""
    try:
        int_value = int(value)
        if value == int_value:
            # This should succeed
            result = ensure_python_int(value)
            assert result == int_value
        else:
            # This should raise TypeError
            with pytest.raises(TypeError):
                ensure_python_int(value)
    except (ValueError, OverflowError):
        # If int() itself fails, ensure_python_int should raise TypeError
        with pytest.raises(TypeError):
            ensure_python_int(value)

if __name__ == "__main__":
    # Run the test with specific examples
    # Note: Can't directly call the function as it's decorated
    # Instead, we'll test the logic directly
    def test_value(value, description):
        print(f"Testing with {description}:")
        try:
            int_value = int(value)
            if value == int_value:
                # This should succeed
                result = ensure_python_int(value)
                print(f"  Result: {result} (expected to succeed)")
            else:
                # This should raise TypeError
                try:
                    ensure_python_int(value)
                    print(f"  ERROR: Should have raised TypeError but got result")
                except TypeError:
                    print(f"  Correctly raised TypeError")
                except Exception as e:
                    print(f"  ERROR: Raised {type(e).__name__} instead of TypeError: {e}")
        except (ValueError, OverflowError) as e:
            # If int() itself fails, ensure_python_int should raise TypeError
            print(f"  int() raises {type(e).__name__}: {e}")
            try:
                ensure_python_int(value)
                print(f"  ERROR: Should have raised TypeError but got result")
            except TypeError:
                print(f"  Correctly raised TypeError after int() failed")
            except Exception as e2:
                print(f"  ERROR: Raised {type(e2).__name__} instead of TypeError: {e2}")

    test_value(float('inf'), "float('inf')")
    test_value(float('-inf'), "float('-inf')")
    test_value(float('nan'), "float('nan')")
    test_value(5.0, "5.0")
    test_value(5.5, "5.5")