#!/usr/bin/env python3
"""Property-based test that discovers the Middleware __repr__ bug."""

from hypothesis import given, strategies as st
from fastapi.middleware import Middleware


class CallableWithoutName:
    """A callable class that doesn't have a __name__ attribute."""
    def __call__(self, app, *args, **kwargs):
        return app


@given(args=st.lists(st.integers(), min_size=1, max_size=3))
def test_middleware_repr_with_unnamed_callable_and_args(args):
    """Test that Middleware.__repr__ doesn't produce invalid syntax with leading commas."""
    callable_without_name = CallableWithoutName()
    middleware = Middleware(callable_without_name, *args)
    repr_str = repr(middleware)

    assert not repr_str.startswith('Middleware(,'), \
        f"Found leading comma in repr: {repr_str}"


if __name__ == "__main__":
    # Run the test to find the bug
    test_middleware_repr_with_unnamed_callable_and_args()