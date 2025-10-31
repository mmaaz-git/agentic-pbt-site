#!/usr/bin/env python3
"""Hypothesis test for FastAPI get_path_param_names empty parameter names bug"""

from hypothesis import given, strategies as st, example
from fastapi.utils import get_path_param_names

@given(st.text())
@example("{}")  # Force it to test with empty braces
@example("/users/{}/posts")
@example("/{}/{}/{}")
def test_get_path_param_names_no_empty_strings(path):
    result = get_path_param_names(path)
    for name in result:
        assert name != '', f"Empty parameter name found for path: {path!r}"

if __name__ == "__main__":
    test_get_path_param_names_no_empty_strings()