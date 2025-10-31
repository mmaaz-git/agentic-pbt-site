#!/usr/bin/env python
"""Property-based test for django.template.Variable trailing dot bug"""

from django.template import Variable, Context
from hypothesis import given, settings, strategies as st, example


@settings(max_examples=100)
@example(n=42)
@example(n=0)
@example(n=123)
@given(st.integers())
def test_variable_numeric_string_with_trailing_dot_should_be_resolvable(n):
    s = str(n) + '.'
    var = Variable(s)

    if var.literal is not None:
        ctx = Context({})
        resolved = var.resolve(ctx)
        assert resolved == var.literal


if __name__ == "__main__":
    test_variable_numeric_string_with_trailing_dot_should_be_resolvable()