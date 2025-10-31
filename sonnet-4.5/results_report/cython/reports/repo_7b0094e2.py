import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case 1: String values
print("Test 1: String values")
template = Template('{{x}}', namespace={'x': 'namespace_value'})
result = template.substitute({'x': 'substitute_value'})
print(f"Result: {result}")
print(f"Expected: substitute_value")
print(f"Actual: {result}")
print()

# Test case 2: Integer values
print("Test 2: Integer values")
template = Template('{{x}}', namespace={'x': 100})
result = template.substitute({'x': 200})
print(f"Result: {result}")
print(f"Expected: 200")
print(f"Actual: {result}")
print()

# Test case 3: Multiple variables
print("Test 3: Multiple variables")
template = Template('{{x}} {{y}}', namespace={'x': 'default_x', 'y': 'default_y'})
result = template.substitute({'x': 'new_x', 'y': 'new_y'})
print(f"Result: {result}")
print(f"Expected: new_x new_y")
print(f"Actual: {result}")