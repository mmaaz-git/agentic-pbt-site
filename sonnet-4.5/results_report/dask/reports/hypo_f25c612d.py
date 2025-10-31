from hypothesis import given, strategies as st
from dask.callbacks import Callback


@given(st.integers(min_value=1, max_value=10))
def test_multiple_callbacks_register_unregister(n):
    callbacks = [Callback() for _ in range(n)]
    initial_active = Callback.active.copy()

    for cb in callbacks:
        cb.register()

    for cb in callbacks:
        assert cb._callback in Callback.active

    for cb in callbacks:
        cb.unregister()

    assert Callback.active == initial_active


if __name__ == "__main__":
    # Run the test
    test_multiple_callbacks_register_unregister()