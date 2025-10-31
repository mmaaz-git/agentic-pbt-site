import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper, loop_pos


# Directly test _compare_group with None
class Item:
    def __init__(self, value):
        self.value = value

item1 = Item(1)
item2 = Item(2)

# Create a loop_pos object
lp = loop_pos([item1, item2], 1)  # Last position

print("Testing _compare_group directly with None:")
print(f"Comparing item with None using '.value' getter...")

try:
    # This should fail according to bug report
    result = lp._compare_group(item2, None, '.value')
    print(f"Result: {result} (Expected AttributeError)")
except AttributeError as e:
    print(f"Got AttributeError as expected: {e}")

# Let's also trace what last_group does
print("\nTracing last_group behavior for last item:")
for loop, item in looper([item1, item2]):
    if loop.last:
        print(f"loop.item: {loop.item}")
        print(f"loop.__next__: {loop.__next__}")
        print(f"Calling _compare_group(item={loop.item}, other={loop.__next__}, getter='.value')")
        try:
            result = loop._compare_group(loop.item, loop.__next__, '.value')
            print(f"Result: {result}")
        except AttributeError as e:
            print(f"AttributeError: {e}")