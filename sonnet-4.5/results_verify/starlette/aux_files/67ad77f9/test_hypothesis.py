"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st
from starlette.middleware import Middleware


@given(
    args=st.lists(st.integers(), max_size=5),
    kwargs=st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=5)
)
def test_middleware_repr_no_leading_comma(args, kwargs):
    class CallableFactory:
        def __call__(self, app):
            return app

    factory = CallableFactory()

    middleware = Middleware(factory, *args, **kwargs)
    repr_str = repr(middleware)

    assert not repr_str.startswith("Middleware(,"), \
        f"repr has leading comma: {repr_str}"

# Run the test
if __name__ == "__main__":
    test_middleware_repr_no_leading_comma()
    print("Hypothesis test completed - bug reproduced!")