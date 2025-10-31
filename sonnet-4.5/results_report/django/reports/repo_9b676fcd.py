import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

# Test 1: Empty source (original bug report)
print("Test 1: Empty source")
print("-" * 40)
exc = MockJinja2Exception(lineno=1, source='', message='test error', filename='test.html')
info = get_exception_info(exc)
print(f"Empty source has {info['total']} lines (expected 0)")
print(f"source_lines: {info['source_lines']}")
print(f"during: '{info['during']}'")
print()

# Test 2: Empty source with lineno > 1 (this should crash)
print("Test 2: Empty source with lineno=2")
print("-" * 40)
try:
    exc2 = MockJinja2Exception(lineno=2, source='', message='test error', filename='test.html')
    info2 = get_exception_info(exc2)
    print(f"Result: total={info2['total']}")
except IndexError as e:
    print(f"ERROR: IndexError occurred: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Whitespace-only source
print("Test 3: Whitespace-only source")
print("-" * 40)
exc3 = MockJinja2Exception(lineno=1, source='   \n  \t  ', message='test error', filename='test.html')
info3 = get_exception_info(exc3)
print(f"Whitespace-only source has {info3['total']} lines (expected 0)")
print(f"source_lines: {info3['source_lines']}")
print(f"during: '{info3['during']}'")