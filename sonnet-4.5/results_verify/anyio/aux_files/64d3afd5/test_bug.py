import sys
sys.path.insert(0, '../../envs/anyio_env')

import anyio
from anyio.streams.stapled import MultiListener

class MockListener:
    def __init__(self, name):
        self.name = name
    async def serve(self, handler, task_group=None):
        pass
    async def aclose(self):
        pass
    @property
    def extra_attributes(self):
        return {}

async def reproduce():
    print("=== Reproducing the bug ===")

    # Create first multi-listener with 2 listeners
    listeners1 = [MockListener("A"), MockListener("B")]
    multi1 = MultiListener(listeners1)

    # Create second multi-listener with 2 listeners
    listeners2 = [MockListener("C"), MockListener("D")]
    multi2 = MultiListener(listeners2)

    print(f"Before combining:")
    print(f"  multi1 has {len(multi1.listeners)} listeners")
    print(f"  multi2 has {len(multi2.listeners)} listeners")

    # Create combined multi-listener from the two multi-listeners
    combined = MultiListener([multi1, multi2])

    print(f"\nAfter combining:")
    print(f"  combined has {len(combined.listeners)} listeners")
    print(f"  multi1 has {len(multi1.listeners)} listeners (CHANGED!)")
    print(f"  multi2 has {len(multi2.listeners)} listeners (CHANGED!)")

    print("\nChecking if multi1 and multi2 are now empty:")
    print(f"  multi1.listeners == []: {multi1.listeners == []}")
    print(f"  multi2.listeners == []: {multi2.listeners == []}")

    # Verify the combined listener has all 4 listeners
    print(f"\nCombined listener contains {len(combined.listeners)} listeners")

    return combined, multi1, multi2

if __name__ == "__main__":
    combined, multi1, multi2 = anyio.run(reproduce)

    print("\n=== Bug confirmed ===")
    print("The nested MultiListener objects were destructively modified.")
    print("Their listeners were 'moved' (deleted) to the new combined MultiListener.")