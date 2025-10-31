import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from fastapi.exceptions import ResponseValidationError


@given(st.lists(st.text(), min_size=1, max_size=1))
def test_single_error_uses_singular_form(errors):
    exc = ResponseValidationError(errors)
    result = str(exc)

    assert "1 validation error:" in result, \
        f"Expected singular 'error', got: {result.split(chr(10))[0]}"

if __name__ == "__main__":
    test_single_error_uses_singular_form()