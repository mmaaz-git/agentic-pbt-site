#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

from anyio.streams.stapled import MultiListener
from hypothesis import given, strategies as st


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(num_inner=st.integers(min_value=1, max_value=5))
def test_multilistener_preserves_nested_listeners(num_inner):
    inner_listeners = [MockListener() for _ in range(num_inner)]
    inner_multi = MultiListener(inner_listeners)

    outer_multi = MultiListener([MockListener(), inner_multi])

    assert len(inner_multi.listeners) == num_inner


# Run the test
if __name__ == "__main__":
    try:
        test_multilistener_preserves_nested_listeners()
    except AssertionError as e:
        print(f"Test failed as expected: {e}")
    except Exception as e:
        print(f"Test errored: {e}")