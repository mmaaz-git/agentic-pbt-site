#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, strategies as st
from fastapi.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


@given(args=st.lists(st.integers(), min_size=1, max_size=3))
def test_middleware_repr_with_unnamed_callable_and_args(args):
    callable_without_name = CallableWithoutName()
    middleware = Middleware(callable_without_name, *args)
    repr_str = repr(middleware)

    print(f"Testing with args={args}: repr={repr_str}")

    assert not repr_str.startswith('Middleware(,'), \
        f"Found leading comma in repr: {repr_str}"


if __name__ == "__main__":
    # Run the test
    test_middleware_repr_with_unnamed_callable_and_args()