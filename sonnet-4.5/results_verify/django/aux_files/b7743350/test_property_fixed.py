import re
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.urls.converters import IntConverter

# More focused test - specifically test negative numbers
converter = IntConverter()
regex = re.compile(f'^{converter.regex}$')

print("Testing specific negative number cases:")
print("=" * 50)

negative_test_cases = ['-1', '-5', '-10', '-100', '-999']

for test_input in negative_test_cases:
    matches_regex = bool(regex.match(test_input))
    print(f"\nInput: '{test_input}'")
    print(f"  Regex '{converter.regex}' matches: {matches_regex}")

    try:
        result = converter.to_python(test_input)
        print(f"  to_python() returned: {result}")
        print(f"  *** INCONSISTENCY: Regex doesn't match but to_python() accepts it ***")
    except (ValueError, TypeError) as e:
        print(f"  to_python() raised: {type(e).__name__}: {e}")
        print(f"  Consistent: Both regex and to_python() reject the input")

print("\n" + "=" * 50)
print("Summary of the issue:")
print("=" * 50)
print("The IntConverter has:")
print(f"  - regex = '{converter.regex}' which matches only non-negative integers")
print(f"  - to_python() method that calls int() without validation")
print("\nThis creates an inconsistency where:")
print("  - The regex rejects negative numbers")
print("  - But to_python() accepts them")
print("\nIn normal Django URL routing this isn't a problem because:")
print("  - The regex is checked FIRST")
print("  - to_python() is only called if regex matches")
print("  - So negative numbers never reach to_python() in normal usage")
print("\nHowever, the inconsistency exists for:")
print("  - Direct/programmatic use of converters")
print("  - Custom converters that extend IntConverter")
print("  - Testing/validation scenarios")