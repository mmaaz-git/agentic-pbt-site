import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template
from Cython.Tempita._tempita import TemplateError

# Now reproduce the exact bug from the report
print("Reproducing the exact bug from the report:")
print("="*50)

# Case 1: {{def }} with space but no function name
print("\n1. Testing {{def }} with space but no function name:")
content1 = "{{def }}{{enddef}}"
try:
    template = Template(content1)
    print("   Result: No error raised - unexpected!")
except TemplateError as e:
    print(f"   Result: TemplateError raised: {e}")
except IndexError as e:
    print(f"   Result: IndexError raised: {e}")
except Exception as e:
    print(f"   Result: {type(e).__name__}: {e}")

# Case 2: {{def}} without space
print("\n2. Testing {{def}} without space:")
content2 = "{{def}}{{enddef}}"
try:
    template = Template(content2)
    print("   Result: No error raised - unexpected!")
except TemplateError as e:
    print(f"   Result: TemplateError raised: {e}")
except IndexError as e:
    print(f"   Result: IndexError raised: {e}")
except Exception as e:
    print(f"   Result: {type(e).__name__}: {e}")

# Case 3: Valid usage with function name
print("\n3. Testing {{def myfunc}} with function name:")
content3 = "{{def myfunc}}Hello{{enddef}}"
try:
    template = Template(content3)
    print("   Result: Template created successfully (as expected)")
except Exception as e:
    print(f"   Result: Error: {e}")