from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


# Create a nested MultiListener with two mock listeners
nested_multi = MultiListener(listeners=[MockListener(), MockListener()])
print(f"Before creating outer MultiListener: {len(nested_multi.listeners)} listeners")

# Create an outer MultiListener that contains the nested one
outer_multi = MultiListener(listeners=[nested_multi, MockListener()])
print(f"After creating outer MultiListener: {len(nested_multi.listeners)} listeners")

# Show the impact
print(f"\nOriginal nested MultiListener now has: {nested_multi.listeners}")
print(f"Outer MultiListener has: {len(outer_multi.listeners)} listeners")