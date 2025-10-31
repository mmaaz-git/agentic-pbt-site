#!/usr/bin/env python3
"""Property-based test for get_authorization_scheme_param bug with multiple spaces."""

from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(st.text(min_size=1), st.text(min_size=1), st.integers(min_value=2, max_value=10))
def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
    assume(" " not in scheme_input)
    assume(" " not in param_input)

    authorization = f"{scheme_input}{' ' * num_spaces}{param_input}"

    scheme, param = get_authorization_scheme_param(authorization)

    assert scheme == scheme_input
    assert param == param_input


if __name__ == "__main__":
    test_get_authorization_scheme_param_multiple_spaces()