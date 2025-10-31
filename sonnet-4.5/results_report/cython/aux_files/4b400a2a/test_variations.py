import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template, TemplateError

# Test various malformed def directives
test_cases = [
    "{{def}}{{enddef}}",
    "{{def }}{{enddef}}",
    "{{def  }}{{enddef}}",
    "{{def\t}}{{enddef}}",
]

for content in test_cases:
    print(f"\nTesting: {repr(content)}")
    try:
        template = Template(content)
        print("  Result: Template created successfully (unexpected!)")
    except TemplateError as e:
        print(f"  Result: TemplateError - {e}")
    except Exception as e:
        print(f"  Result: {type(e).__name__} - {e}")