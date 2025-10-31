import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

print("Testing SyntaxError vs NameError position info:")
print("=" * 50)

# Test SyntaxError
content = "Line 1\nLine 2\n{{/}}"
template = Template(content)

try:
    template.substitute({})
except SyntaxError as e:
    print(f"SyntaxError: {e}")

# Test NameError for comparison
content2 = "Line 1\nLine 2\n{{undefined}}"
template2 = Template(content2)

try:
    template2.substitute({})
except NameError as e:
    print(f"NameError: {e}")

print("\n" + "=" * 50)
print("Testing other exception types:")

# Test ValueError
content3 = "Line 1\nLine 2\n{{int('abc')}}"
template3 = Template(content3)

try:
    template3.substitute({})
except ValueError as e:
    print(f"ValueError: {e}")

# Test TypeError
content4 = "Line 1\nLine 2\n{{len(123)}}"
template4 = Template(content4)

try:
    template4.substitute({})
except TypeError as e:
    print(f"TypeError: {e}")