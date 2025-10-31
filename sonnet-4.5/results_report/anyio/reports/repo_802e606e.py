from anyio.streams.stapled import MultiListener


class MockListener:
    @property
    def extra_attributes(self):
        return {}


listener = MockListener()
nested = MultiListener(listeners=[listener])
print(f"Before: nested.listeners = {nested.listeners}")

flat = MultiListener(listeners=[nested])
print(f"After: nested.listeners = {nested.listeners}")
