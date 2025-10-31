import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Fix the previous test - need to pass the value
content2 = """
{{def greet(name)}}
Hello {{name}}!
{{enddef}}

{{greet(name='World')}}
"""

try:
    template = Template(content2)
    result = template.substitute({})
    print(f"Example with keyword arg works: {result.strip()}")
except Exception as e:
    print(f"Example failed: {e}")

# Now test {{def}} without function name
print("\nTesting {{def}} without function name:")
content3 = "{{def}}{{enddef}}"
try:
    template = Template(content3)
    print("No error raised!")
except TemplateError as e:
    print(f"TemplateError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Test {{def }} with space but no function name
print("\nTesting {{def }} with space but no function name:")
content4 = "{{def }}{{enddef}}"
try:
    template = Template(content4)
    print("No error raised!")
except TemplateError as e:
    print(f"TemplateError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")