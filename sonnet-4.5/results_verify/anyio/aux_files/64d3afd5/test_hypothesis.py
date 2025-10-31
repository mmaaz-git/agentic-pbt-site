import sys
sys.path.insert(0, '../../envs/anyio_env')

from hypothesis import given, strategies as st
from anyio.streams.stapled import MultiListener

class MockListener:
    def __init__(self, name):
        self.name = name
    async def serve(self, handler, task_group=None):
        pass
    async def aclose(self):
        pass
    @property
    def extra_attributes(self):
        return {}

@given(st.integers(min_value=1, max_value=5))
def test_multi_listener_flattening(n_listeners):
    listeners1 = [MockListener(f"listener_{i}") for i in range(n_listeners)]
    multi1 = MultiListener(listeners1)
    original_multi1_count = len(multi1.listeners)

    listeners2 = [MockListener(f"listener_{i + n_listeners}") for i in range(n_listeners)]
    multi2 = MultiListener(listeners2)
    original_multi2_count = len(multi2.listeners)

    # Create combined MultiListener
    combined = MultiListener([multi1, multi2])

    # Check the assertions from the bug report
    print(f"\nTest with n_listeners={n_listeners}:")
    print(f"  combined.listeners has {len(combined.listeners)} items (expected {2 * n_listeners})")
    print(f"  multi1.listeners has {len(multi1.listeners)} items (was {original_multi1_count})")
    print(f"  multi2.listeners has {len(multi2.listeners)} items (was {original_multi2_count})")

    assert len(combined.listeners) == 2 * n_listeners, f"Combined should have {2*n_listeners} listeners"
    assert len(multi1.listeners) == 0, f"multi1 should be empty after being nested"
    assert len(multi2.listeners) == 0, f"multi2 should be empty after being nested"

if __name__ == "__main__":
    print("Running hypothesis test...")
    test_multi_listener_flattening()
    print("\nAll tests passed - the behavior is consistent across different values.")