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

multi1 = MultiListener(listeners=[listener1, listener2])
print(f"multi1.listeners before: {len(multi1.listeners)} listeners")
print(f"  Contents: {multi1.listeners}")

multi2 = MultiListener(listeners=[multi1])
print(f"multi1.listeners after creating multi2: {len(multi1.listeners)} listeners")
print(f"  Contents: {multi1.listeners}")

print(f"multi2.listeners: {len(multi2.listeners)} listeners")
print(f"  Contents: {multi2.listeners}")

assert len(multi1.listeners) == 0, "Bug confirmed: multi1.listeners was emptied"