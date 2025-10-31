import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import parse_signature
from Cython.Tempita import Template

# Test case 1: single argument
result = parse_signature("name", "test", (1, 1))
sig_args, _, _, _ = result
print(f"Signature 'name' parsed as: {sig_args}")

# Test case 2: multiple arguments
result2 = parse_signature("name, greeting", "test", (1, 1))
sig_args2, _, _, _ = result2
print(f"Signature 'name, greeting' parsed as: {sig_args2}")

# Test case 3: Template usage
try:
    content = "{{def greet(name)}}Hello, {{name}}!{{enddef}}{{greet('World')}}"
    template = Template(content)
    result = template.substitute({})
    print(f"Template result: {result}")
except Exception as e:
    print(f"Template error: {type(e).__name__}: {e}")