from hypothesis import given, strategies as st
from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.integers(min_value=1, max_value=10))
def test_multilistener_does_not_mutate_inputs(num_listeners: int):
    listeners = [MockListener() for _ in range(num_listeners)]
    nested = MultiListener(listeners=listeners)
    original_count = len(nested.listeners)

    MultiListener(listeners=[nested, MockListener()])

    assert len(nested.listeners) == original_count, \
        f"MultiListener mutated input: expected {original_count} listeners, but found {len(nested.listeners)}"

if __name__ == "__main__":
    # Run the test directly
    test_multilistener_does_not_mutate_inputs()