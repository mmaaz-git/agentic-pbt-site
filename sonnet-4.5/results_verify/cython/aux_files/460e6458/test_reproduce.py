import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper


class Item:
    def __init__(self, value):
        self.value = value


items = [Item(1), Item(2), Item(3)]

print("Testing last_group with attribute getter on last item:")
for loop, item in looper(items):
    if loop.last:
        try:
            result = loop.last_group('.value')
            print(f"Result: {result}")
        except AttributeError as e:
            print(f"AttributeError: {e}")

print("\nTesting first_group with attribute getter on first item:")
for loop, item in looper(items):
    if loop.first:
        try:
            result = loop.first_group('.value')
            print(f"Result: {result}")
        except AttributeError as e:
            print(f"AttributeError: {e}")