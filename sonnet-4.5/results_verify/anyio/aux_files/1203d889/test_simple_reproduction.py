from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


nested_multi = MultiListener(listeners=[MockListener(), MockListener()])
print(f"Before: {len(nested_multi.listeners)} listeners")

outer_multi = MultiListener(listeners=[nested_multi, MockListener()])
print(f"After: {len(nested_multi.listeners)} listeners")