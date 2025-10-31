import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

print("Test 1: is_same_domain('A', 'A')")
print(f"Result: {is_same_domain('A', 'A')}")
print(f"Expected: True\n")

print("Test 2: is_same_domain('a', 'a')")
print(f"Result: {is_same_domain('a', 'a')}")
print(f"Expected: True\n")

print("Test 3: is_same_domain('Example.COM', 'Example.COM')")
print(f"Result: {is_same_domain('Example.COM', 'Example.COM')}")
print(f"Expected: True\n")

print("Test 4: is_same_domain('example.com', 'example.com')")
print(f"Result: {is_same_domain('example.com', 'example.com')}")
print(f"Expected: True\n")

print("Test 5: is_same_domain('Example.com', 'example.com')")
print(f"Result: {is_same_domain('Example.com', 'example.com')}")
print(f"Expected: True (domains are case-insensitive)\n")

print("Test 6: is_same_domain('EXAMPLE.COM', 'example.com')")
print(f"Result: {is_same_domain('EXAMPLE.COM', 'example.com')}")
print(f"Expected: True (domains are case-insensitive)\n")

print("Final assertion test:")
try:
    assert is_same_domain('A', 'A'), "Same domain with same case should match!"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")