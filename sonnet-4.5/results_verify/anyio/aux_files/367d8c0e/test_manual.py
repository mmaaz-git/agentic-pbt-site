from anyio.streams.stapled import MultiListener


class MockListener:
    def __init__(self, name):
        self.name = name

    @property
    def extra_attributes(self):
        return {}


# Manual test with num_listeners=1
def test_with_one_listener():
    listeners = [MockListener(f"listener_0")]
    nested = MultiListener(listeners=listeners)

    original_count = len(nested.listeners)
    print(f"Original count of nested.listeners: {original_count}")
    assert original_count == 1

    flat = MultiListener(listeners=[nested])
    print(f"After creating flat MultiListener, nested.listeners count: {len(nested.listeners)}")

    assert len(nested.listeners) == original_count, f"Expected {original_count} listeners, but found {len(nested.listeners)}"


if __name__ == "__main__":
    print("Testing with num_listeners=1...")
    try:
        test_with_one_listener()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")