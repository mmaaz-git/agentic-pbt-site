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


listener1 = MockListener("listener1")
listener2 = MockListener("listener2")

# Create multi1 with two listeners
multi1 = MultiListener(listeners=[listener1, listener2])
print(f"multi1.listeners before creating multi2: {multi1.listeners}")
print(f"multi1.listeners length before: {len(multi1.listeners)}")

# Create multi2 with multi1 as input
multi2 = MultiListener(listeners=[multi1])
print(f"multi1.listeners after creating multi2: {multi1.listeners}")
print(f"multi1.listeners length after: {len(multi1.listeners)}")
print(f"multi2.listeners: {multi2.listeners}")
print(f"multi2.listeners length: {len(multi2.listeners)}")

# This assertion demonstrates the bug
assert len(multi1.listeners) == 0, f"multi1.listeners was mutated to empty! Length is {len(multi1.listeners)}"
print("\nBUG CONFIRMED: multi1.listeners was emptied when passed to multi2!")