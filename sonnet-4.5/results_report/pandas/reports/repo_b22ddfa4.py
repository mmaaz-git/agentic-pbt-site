import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import parse_signature
from Cython.Tempita import Template

# Test 1: Single argument without default
print("Test 1: Single argument without default")
result = parse_signature("name", "test", (1, 1))
sig_args, var_arg, var_kw, defaults = result
print(f"  Input: 'name'")
print(f"  Parsed sig_args: {sig_args}")
print(f"  Expected: ['name']")
print(f"  Result: {'FAIL' if sig_args != ['name'] else 'PASS'}")
print()

# Test 2: Multiple arguments without defaults
print("Test 2: Multiple arguments without defaults")
result2 = parse_signature("name, greeting", "test", (1, 1))
sig_args2, var_arg2, var_kw2, defaults2 = result2
print(f"  Input: 'name, greeting'")
print(f"  Parsed sig_args: {sig_args2}")
print(f"  Expected: ['name', 'greeting']")
print(f"  Result: {'FAIL' if sig_args2 != ['name', 'greeting'] else 'PASS'}")
print()

# Test 3: Template usage with function definition
print("Test 3: Template usage with function definition")
content = "{{def greet(name)}}Hello, {{name}}!{{enddef}}{{greet('World')}}"
try:
    template = Template(content)
    result = template.substitute({})
    print(f"  Template output: {result}")
    print(f"  Result: PASS")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
    print(f"  Result: FAIL")
print()

# Test 4: Arguments with defaults (should work)
print("Test 4: Arguments with defaults (should work)")
result3 = parse_signature("name='default'", "test", (1, 1))
sig_args3, var_arg3, var_kw3, defaults3 = result3
print(f"  Input: \"name='default'\"")
print(f"  Parsed sig_args: {sig_args3}")
print(f"  Defaults: {defaults3}")
print(f"  Expected: sig_args=['name'], defaults with 'name' key")
print(f"  Result: {'PASS' if sig_args3 == ['name'] and 'name' in defaults3 else 'FAIL'}")