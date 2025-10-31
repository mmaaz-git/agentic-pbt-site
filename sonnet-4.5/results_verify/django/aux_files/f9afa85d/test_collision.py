from django.core.cache.utils import make_template_fragment_key

# Test the collision case
key1 = make_template_fragment_key('fragment', ['a', 'b'])
key2 = make_template_fragment_key('fragment', ['a:b'])

print(f"['a', 'b'] produces: {key1}")
print(f"['a:b'] produces: {key2}")
print(f"Collision: {key1 == key2}")
print()

# Additional test cases
key3 = make_template_fragment_key('fragment', ['a', 'b', 'c'])
key4 = make_template_fragment_key('fragment', ['a:b', 'c'])
key5 = make_template_fragment_key('fragment', ['a', 'b:c'])
key6 = make_template_fragment_key('fragment', ['a:b:c'])

print(f"['a', 'b', 'c'] produces: {key3}")
print(f"['a:b', 'c'] produces: {key4}")
print(f"['a', 'b:c'] produces: {key5}")
print(f"['a:b:c'] produces: {key6}")
print(f"Collision ['a', 'b', 'c'] == ['a:b', 'c']: {key3 == key4}")
print(f"Collision ['a', 'b', 'c'] == ['a', 'b:c']: {key3 == key5}")
print(f"Collision ['a', 'b', 'c'] == ['a:b:c']: {key3 == key6}")