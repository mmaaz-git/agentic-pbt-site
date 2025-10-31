import attrs
from attrs import cmp_using
import pytest
from hypothesis import given, strategies as st

@given(st.just(None))  # Using a simple strategy since we're testing error messages
def test_cmp_using_error_message_typo(dummy):
    """Test that cmp_using raises ValueError with typo in error message when eq is missing."""
    with pytest.raises(ValueError, match="eq must be define"):
        cmp_using(lt=lambda a, b: a < b)

if __name__ == "__main__":
    # Run the test to show it catches the typo
    from hypothesis import find
    try:
        # Use hypothesis to find an example
        find(st.just(None), lambda x: test_cmp_using_error_message_typo() or True)
    except:
        # Just run the test directly
        try:
            cmp_using(lt=lambda a, b: a < b)
        except ValueError as e:
            print(f"ValueError raised with message: {e}")
            if "eq must be define" in str(e):
                print("Test confirmed: The typo 'eq must be define' exists in the error message")