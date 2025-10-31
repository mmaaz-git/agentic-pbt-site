from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


inner_listeners = [MockListener()]
inner_multi = MultiListener(listeners=inner_listeners)

print(f"Before: len(inner_multi.listeners) = {len(inner_multi.listeners)}")

outer_multi = MultiListener(listeners=[inner_multi])

print(f"After: len(inner_multi.listeners) = {len(inner_multi.listeners)}")

print(f"\nExpected: Both should show 1")
print(f"Actual: First shows 1, second shows {len(inner_multi.listeners)}")