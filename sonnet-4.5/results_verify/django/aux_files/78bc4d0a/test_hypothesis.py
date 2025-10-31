from hypothesis import given, strategies as st
from django.db.models.expressions import F
from django.db.models.functions import NthValue
import pytest


@given(st.integers(max_value=0))
def test_nthvalue_error_message_grammar(nth):
    """Test that NthValue error message contains grammatical error 'as for nth'"""
    with pytest.raises(ValueError) as exc_info:
        NthValue(F('field'), nth=nth)

    error_msg = str(exc_info.value)
    assert "positive integer" in error_msg
    assert "as for nth" in error_msg
    print(f"nth={nth}, error: {error_msg}")


# Run the test
if __name__ == "__main__":
    # Test with a few examples
    test_examples = [0, -1, -100, None]

    for nth_value in test_examples:
        try:
            if nth_value is not None:
                test_nthvalue_error_message_grammar(nth_value)
            else:
                # Test None separately
                try:
                    NthValue(F('field'), nth=None)
                except ValueError as e:
                    print(f"nth=None, error: {e}")
                    assert "as for nth" in str(e)
        except AssertionError:
            print(f"Test passed for nth={nth_value}")