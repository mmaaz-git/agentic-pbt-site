import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test SyntaxError - missing position info
content = "Line 1\nLine 2\n{{/}}"
template = Template(content)

print("Testing SyntaxError with template: {{/}}")
print("-" * 50)
try:
    template.substitute({})
except SyntaxError as e:
    print(f"SyntaxError: {e}")
print()

# Test NameError - has position info
content2 = "Line 1\nLine 2\n{{undefined}}"
template2 = Template(content2)

print("Testing NameError with template: {{undefined}}")
print("-" * 50)
try:
    template2.substitute({})
except NameError as e:
    print(f"NameError: {e}")
print()

# Test ValueError - has position info
content3 = "Line 1\nLine 2\n{{int('abc')}}"
template3 = Template(content3)

print("Testing ValueError with template: {{int('abc')}}")
print("-" * 50)
try:
    template3.substitute({})
except ValueError as e:
    print(f"ValueError: {e}")
print()

# Test TypeError - has position info
content4 = "Line 1\nLine 2\n{{len(123)}}"
template4 = Template(content4)

print("Testing TypeError with template: {{len(123)}}")
print("-" * 50)
try:
    template4.substitute({})
except TypeError as e:
    print(f"TypeError: {e}")