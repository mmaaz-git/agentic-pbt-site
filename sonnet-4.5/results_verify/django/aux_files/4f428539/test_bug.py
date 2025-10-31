import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.core.management.utils import handle_extensions

print("Test case 1: Empty string")
result = handle_extensions([''])
print(f"Result: {result}")
print(f"'.' in result: {'.' in result}")
assert '.' in result

print("\nTest case 2: Double comma")
result = handle_extensions(['html,,css'])
print(f"Result: {result}")
print(f"'.' in result: {'.' in result}")
assert '.' in result

print("\nTest case 3: Trailing comma")
result = handle_extensions(['html,'])
print(f"Result: {result}")
print(f"'.' in result: {'.' in result}")
assert '.' in result

print("\nTest case 4: Leading comma")
result = handle_extensions([',html'])
print(f"Result: {result}")
print(f"'.' in result: {'.' in result}")
assert '.' in result

print("\nAll tests passed - bug confirmed!")