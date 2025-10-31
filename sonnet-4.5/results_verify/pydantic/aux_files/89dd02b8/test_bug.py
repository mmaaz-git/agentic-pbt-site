#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


# First test: simple reproduction
inner_multi = MultiListener([MockListener(), MockListener()])
print(f"Inner listeners before: {len(inner_multi.listeners)}")

outer_multi = MultiListener([MockListener(), inner_multi])
print(f"Inner listeners after: {len(inner_multi.listeners)}")

# Second test: check the outer listener
print(f"Outer listeners count: {len(outer_multi.listeners)}")
print(f"Outer listener contents: {outer_multi.listeners}")

# Third test: hypothesis-style test with different num_inner values
for num_inner in range(1, 6):
    inner_listeners = [MockListener() for _ in range(num_inner)]
    inner_multi = MultiListener(inner_listeners)

    outer_multi = MultiListener([MockListener(), inner_multi])

    print(f"Test with {num_inner} inner listeners - Inner after nesting: {len(inner_multi.listeners)}")
    assert len(inner_multi.listeners) == 0, f"Expected 0 listeners after nesting, got {len(inner_multi.listeners)}"
    print(f"  (Assertion would fail if we expected {num_inner} listeners)")