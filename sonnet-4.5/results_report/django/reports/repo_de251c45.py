import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.cache.utils import make_template_fragment_key

# Test case 1: ["a:", "b"] vs ["a", ":b"]
key1 = make_template_fragment_key("fragment", ["a:", "b"])
key2 = make_template_fragment_key("fragment", ["a", ":b"])

print("Test case 1:")
print(f"vary_on=['a:', 'b'] produces key: {key1}")
print(f"vary_on=['a', ':b'] produces key: {key2}")
print(f"Keys are equal: {key1 == key2}")
print()

# Test case 2: ["x:y", "z"] vs ["x", "y:z"]
key3 = make_template_fragment_key("test", ["x:y", "z"])
key4 = make_template_fragment_key("test", ["x", "y:z"])

print("Test case 2:")
print(f"vary_on=['x:y', 'z'] produces key: {key3}")
print(f"vary_on=['x', 'y:z'] produces key: {key4}")
print(f"Keys are equal: {key3 == key4}")