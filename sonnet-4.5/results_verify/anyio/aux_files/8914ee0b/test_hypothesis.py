import anyio
from hypothesis import given, settings, strategies as st
from anyio.streams.stapled import MultiListener


class MockListener:
    async def serve(self, handler, task_group=None):
        pass

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=500)
def test_multilistener_doesnt_mutate_input(num_listeners):
    async def test():
        inner_listeners = [MockListener() for _ in range(num_listeners)]
        inner_multi = MultiListener(listeners=inner_listeners)

        original_count = len(inner_multi.listeners)

        outer_multi = MultiListener(listeners=[inner_multi])

        after_count = len(inner_multi.listeners)
        assert after_count == original_count, f"Creating outer MultiListener mutated inner: {original_count} -> {after_count}"

    anyio.run(test)

if __name__ == "__main__":
    # Run the hypothesis test
    test_multilistener_doesnt_mutate_input()
    print("All hypothesis tests passed")