from django.core.cache.utils import make_template_fragment_key

# Test case showing the collision
key1 = make_template_fragment_key('fragment', ['a', 'b'])
key2 = make_template_fragment_key('fragment', ['a:b'])

print(f"['a', 'b'] produces: {key1}")
print(f"['a:b'] produces: {key2}")
print(f"Collision: {key1 == key2}")

# Additional test cases showing more collisions
key3 = make_template_fragment_key('test', ['x', 'y', 'z'])
key4 = make_template_fragment_key('test', ['x:y', 'z'])
key5 = make_template_fragment_key('test', ['x', 'y:z'])
key6 = make_template_fragment_key('test', ['x:y:z'])

print(f"\n['x', 'y', 'z'] produces: {key3}")
print(f"['x:y', 'z'] produces: {key4}")
print(f"['x', 'y:z'] produces: {key5}")
print(f"['x:y:z'] produces: {key6}")
print(f"All four keys equal: {key3 == key4 == key5 == key6}")