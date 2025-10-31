import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test valid {{def}} usage
print("Valid {{def}} usage examples:")

# Example 1: def with function name and no arguments
content1 = """
{{def myfunc}}
Hello from function
{{enddef}}

{{myfunc}}
"""

try:
    template = Template(content1)
    result = template.substitute({})
    print(f"Example 1 works: {result.strip()}")
except Exception as e:
    print(f"Example 1 failed: {e}")

# Example 2: def with function name and arguments
content2 = """
{{def greet(name)}}
Hello {{name}}!
{{enddef}}

{{greet('World')}}
"""

try:
    template = Template(content2)
    result = template.substitute({})
    print(f"Example 2 works: {result.strip()}")
except Exception as e:
    print(f"Example 2 failed: {e}")