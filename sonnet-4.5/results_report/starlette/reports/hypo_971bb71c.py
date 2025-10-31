from hypothesis import given, strategies as st
from starlette.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


@given(
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=3)
)
def test_middleware_repr_no_leading_comma(args, kwargs):
    callable_obj = CallableWithoutName()
    middleware = Middleware(callable_obj, *args, **kwargs)
    repr_str = repr(middleware)

    if args or kwargs:
        assert not repr_str.split("(", 1)[1].startswith(", "), \
            f"repr should not have leading comma: {repr_str}"


# Run the test with the specific failing input
if __name__ == "__main__":
    # Call the underlying function directly
    callable_obj = CallableWithoutName()
    middleware = Middleware(callable_obj, 0)
    repr_str = repr(middleware)
    print(f"Test with args=[0], kwargs={{}}")
    print(f"Result: {repr_str}")
    if [0] or {}:
        if repr_str.split("(", 1)[1].startswith(", "):
            print(f"FAILED: repr should not have leading comma: {repr_str}")
        else:
            print("PASSED")