import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper


class Item:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


items = [Item(1), Item(2), Item(3)]

print("Testing different getter patterns:\n")

# Test 1: Attribute getter on last item
print("1. Testing last_group('.value') on last item:")
for loop, item in looper(items):
    if loop.last:
        print(f"  Item value: {item.value}")
        print(f"  loop.__next__: {loop.__next__}")
        try:
            result = loop.last_group('.value')
            print(f"  Result: {result}")
        except AttributeError as e:
            print(f"  AttributeError: {e}")

# Test 2: Method getter on last item
print("\n2. Testing last_group('.get_value()') on last item:")
for loop, item in looper(items):
    if loop.last:
        try:
            result = loop.last_group('.get_value()')
            print(f"  Result: {result}")
        except AttributeError as e:
            print(f"  AttributeError: {e}")

# Test 3: First group with attribute getter
print("\n3. Testing first_group('.value') on first item:")
for loop, item in looper(items):
    if loop.first:
        print(f"  Item value: {item.value}")
        print(f"  loop.previous: {loop.previous}")
        try:
            result = loop.first_group('.value')
            print(f"  Result: {result}")
        except AttributeError as e:
            print(f"  AttributeError: {e}")

# Test 4: Dict key getter
print("\n4. Testing with dict key getter:")
dict_items = [{'val': 1}, {'val': 2}, {'val': 3}]
for loop, item in looper(dict_items):
    if loop.last:
        print(f"  Item: {item}")
        print(f"  loop.__next__: {loop.__next__}")
        try:
            result = loop.last_group('val')
            print(f"  Result: {result}")
        except (AttributeError, TypeError) as e:
            print(f"  Error: {e}")

# Test 5: Callable getter
print("\n5. Testing with callable getter:")
def value_getter(obj):
    return obj.value if hasattr(obj, 'value') else None

for loop, item in looper(items):
    if loop.last:
        try:
            result = loop.last_group(value_getter)
            print(f"  Result: {result}")
        except AttributeError as e:
            print(f"  AttributeError: {e}")