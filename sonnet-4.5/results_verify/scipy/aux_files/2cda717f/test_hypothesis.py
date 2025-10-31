from hypothesis import given, strategies as st

@given(st.booleans())
def test_main_handles_missing_pooch(pooch_installed):
    """Property: main() should give clear error when pooch is missing, not crash."""
    import sys
    from unittest.mock import patch

    pooch_value = None if not pooch_installed else __import__('pooch')

    with patch.dict('sys.modules', {'pooch': pooch_value}):
        try:
            from scipy.datasets._download_all import main
            if not pooch_installed:
                assert False, "Should have crashed or raised ImportError"
        except (ImportError, AttributeError) as e:
            if not pooch_installed and isinstance(e, AttributeError):
                assert False, f"Wrong error type: {e}"

# Run the test
test_main_handles_missing_pooch()