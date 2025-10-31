from hypothesis import given, strategies as st
from Cython.Plex.Actions import Call

@given(st.integers())
def test_call_repr_with_callable_object(value):
    class MyCallable:
        def __call__(self, scanner, text):
            return value

    action = Call(MyCallable())
    repr_str = repr(action)
    assert 'Call' in repr_str

# Run the test
if __name__ == "__main__":
    test_call_repr_with_callable_object()