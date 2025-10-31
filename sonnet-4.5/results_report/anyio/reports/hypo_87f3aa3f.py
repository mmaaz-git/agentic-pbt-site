from hypothesis import given, strategies as st
from anyio.streams.stapled import MultiListener
from dataclasses import dataclass


@dataclass
class MockListener:
    name: str

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.integers(min_value=1, max_value=10))
def test_multilistener_does_not_mutate_input(num_listeners):
    listeners = [MockListener(f"listener{i}") for i in range(num_listeners)]
    multi1 = MultiListener(listeners=listeners)

    original_count = len(multi1.listeners)
    assert original_count == num_listeners

    multi2 = MultiListener(listeners=[multi1])

    # This assertion should pass but fails due to the bug
    assert len(multi1.listeners) == original_count, f"Expected {original_count} listeners in multi1, but found {len(multi1.listeners)}"


if __name__ == "__main__":
    test_multilistener_does_not_mutate_input()