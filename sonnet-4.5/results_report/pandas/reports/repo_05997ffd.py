import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case 1: Basic mutation with namespace
user_dict = {'x': 'value', 'y': 42}
print("Test 1: Basic mutation with namespace")
print(f"Before: {user_dict}")

template = Template('{{x}}', namespace={'z': 100})
result = template.substitute(user_dict)

print(f"After:  {user_dict}")
print(f"Result: {result}")
print(f"Added keys: {set(user_dict.keys()) - {'x', 'y'}}")
print()

# Test case 2: Mutation without namespace
user_dict2 = {'a': 1, 'b': 2}
print("Test 2: Mutation without namespace")
print(f"Before: {user_dict2}")

template2 = Template('{{a}} {{b}}')
result2 = template2.substitute(user_dict2)

print(f"After:  {user_dict2}")
print(f"Result: {result2}")
print(f"Added keys: {set(user_dict2.keys()) - {'a', 'b'}}")
