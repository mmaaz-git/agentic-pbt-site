import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper


class Item:
    def __init__(self, value):
        self.value = value


items = [Item(1), Item(2), Item(3)]

print("Looking at last_group implementation:")
for loop, item in looper(items):
    if loop.last:
        print(f"loop.last = {loop.last}")
        print("In last_group, line 136-137 checks: if self.last: return True")
        print("So it returns True immediately without calling _compare_group!")
        result = loop.last_group('.value')
        print(f"Result: {result}")

print("\n\nBUT if we had items with same values at the end:")
items2 = [Item(1), Item(2), Item(2)]
for i, (loop, item) in enumerate(looper(items2)):
    print(f"Item {i}: value={item.value}, last={loop.last}, last_group={loop.last_group('.value')}")