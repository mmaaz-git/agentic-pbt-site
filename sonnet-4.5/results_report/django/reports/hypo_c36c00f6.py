from hypothesis import given, strategies as st, example
from django.db.models.expressions import F
from django.db.models.functions import NthValue
import pytest


@given(st.integers(max_value=0))
@example(0)  # Ensure we test with 0
@example(-1)  # Ensure we test with -1
def test_nthvalue_error_message_grammar(nth):
    """Test that NthValue raises ValueError with grammatically incorrect message."""
    with pytest.raises(ValueError) as exc_info:
        NthValue(F('field'), nth=nth)

    error_msg = str(exc_info.value)
    assert "positive integer" in error_msg
    # This assertion demonstrates the bug - "as for nth" is grammatically incorrect
    assert "as for nth" in error_msg