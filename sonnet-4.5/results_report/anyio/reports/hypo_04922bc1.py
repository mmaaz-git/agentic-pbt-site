import hypothesis
from anyio.streams.stapled import MultiListener


class MockListener:
    def __init__(self, name: str):
        self.name = name

    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@hypothesis.given(hypothesis.strategies.just(None))
def test_multilistener_should_not_mutate_nested(dummy):
    listener1 = MockListener("A")
    listener2 = MockListener("B")

    nested_multi = MultiListener([listener1, listener2])
    original_length = len(nested_multi.listeners)

    outer_multi = MultiListener([nested_multi])

    assert len(nested_multi.listeners) == original_length, \
        f"Expected {original_length} listeners, but found {len(nested_multi.listeners)}"


if __name__ == "__main__":
    test_multilistener_should_not_mutate_nested()