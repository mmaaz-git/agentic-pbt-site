from hypothesis import given, strategies as st
from django.db.models.expressions import F
from django.db.models.functions import NthValue


@given(st.integers(max_value=0))
def test_nthvalue_error_message_grammar(nth):
    """Test that NthValue error message contains grammatical error 'as for nth'"""
    try:
        NthValue(F('field'), nth=nth)
        assert False, f"Should have raised ValueError for nth={nth}"
    except ValueError as e:
        error_msg = str(e)
        assert "positive integer" in error_msg, f"Error doesn't mention 'positive integer': {error_msg}"
        assert "as for nth" in error_msg, f"Error doesn't contain grammatical error 'as for nth': {error_msg}"
        return True


# Test without Hypothesis
def simple_test():
    test_examples = [0, -1, -100, None]

    for nth_value in test_examples:
        try:
            NthValue(F('field'), nth=nth_value)
            print(f"ERROR: Should have raised ValueError for nth={nth_value}")
        except ValueError as e:
            error_msg = str(e)
            print(f"nth={nth_value}, error: {error_msg}")
            assert "positive integer" in error_msg
            assert "as for nth" in error_msg
            print(f"✓ Test passed for nth={nth_value}")


if __name__ == "__main__":
    print("Running simple tests:")
    simple_test()

    print("\nRunning Hypothesis test:")
    # Run the property test manually with some values
    for i in range(10):
        nth_val = -i
        result = test_nthvalue_error_message_grammar(nth_val)
        if result:
            print(f"✓ Hypothesis test passed for nth={nth_val}")