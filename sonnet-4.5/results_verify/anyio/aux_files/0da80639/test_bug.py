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


def test_multilistener_should_not_mutate_nested():
    listener1 = MockListener("A")
    listener2 = MockListener("B")

    nested_multi = MultiListener([listener1, listener2])
    original_length = len(nested_multi.listeners)

    outer_multi = MultiListener([nested_multi])

    assert len(nested_multi.listeners) == original_length, f"Expected {original_length} listeners, but got {len(nested_multi.listeners)}"


# Reproducing the Bug
listener1 = MockListener("A")
listener2 = MockListener("B")

nested_multi = MultiListener([listener1, listener2])
print(f"Before: nested_multi has {len(nested_multi.listeners)} listeners")

outer_multi = MultiListener([nested_multi])
print(f"After: nested_multi has {len(nested_multi.listeners)} listeners")
if len(nested_multi.listeners) == 0:
    print(f"Bug: nested_multi.listeners was cleared!")

# Run the test
try:
    test_multilistener_should_not_mutate_nested()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")