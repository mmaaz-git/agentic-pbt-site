from anyio.streams.stapled import MultiListener
from hypothesis import given, strategies as st


class MockListener:
    def __init__(self, name):
        self.name = name

    @property
    def extra_attributes(self):
        return {}


@given(num_listeners=st.integers(min_value=1, max_value=5))
def test_multilistener_does_not_mutate_nested(num_listeners):
    listeners = [MockListener(f"listener_{i}") for i in range(num_listeners)]
    nested = MultiListener(listeners=listeners)

    original_count = len(nested.listeners)
    assert original_count == num_listeners

    flat = MultiListener(listeners=[nested])

    assert len(nested.listeners) == original_count


# Run the test
if __name__ == "__main__":
    test_multilistener_does_not_mutate_nested()