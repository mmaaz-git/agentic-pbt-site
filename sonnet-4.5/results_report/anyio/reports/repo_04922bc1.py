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


listener1 = MockListener("A")
listener2 = MockListener("B")

nested_multi = MultiListener([listener1, listener2])
print(f"Before: nested_multi has {len(nested_multi.listeners)} listeners")

outer_multi = MultiListener([nested_multi])
print(f"After: nested_multi has {len(nested_multi.listeners)} listeners")
print(f"Bug: nested_multi.listeners was cleared!")