import re
import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Test 1: Basic reproduction of the bug
print("=" * 50)
print("Test 1: Basic Bug Reproduction")
print("=" * 50)

from django.urls.converters import IntConverter

converter = IntConverter()
regex = re.compile(f'^{converter.regex}$')

# Test negative input
negative_input = '-5'
print(f"Testing input: '{negative_input}'")
print(f"Regex pattern: {converter.regex}")
print(f"Does regex match '{negative_input}'? {bool(regex.match(negative_input))}")

# Call to_python on negative input
try:
    result = converter.to_python(negative_input)
    print(f"to_python('{negative_input}') = {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"to_python('{negative_input}') raised: {type(e).__name__}: {e}")

# Test positive input for comparison
print("\n" + "-" * 30)
positive_input = '5'
print(f"Testing input: '{positive_input}'")
print(f"Does regex match '{positive_input}'? {bool(regex.match(positive_input))}")

try:
    result = converter.to_python(positive_input)
    print(f"to_python('{positive_input}') = {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"to_python('{positive_input}') raised: {type(e).__name__}: {e}")

# Test 2: Property-based test
print("\n" + "=" * 50)
print("Test 2: Property-Based Test")
print("=" * 50)

from hypothesis import given, strategies as st, settings
import pytest

test_cases_checked = []
failures = []

@given(st.text(min_size=1, max_size=20))
@settings(max_examples=100, deadline=None)
def test_int_converter_contract_to_python_validates_regex(s):
    converter = IntConverter()
    regex = re.compile(f'^{converter.regex}$')

    test_cases_checked.append(s)

    if regex.match(s):
        try:
            result = converter.to_python(s)
            if not isinstance(result, int):
                failures.append(f"Input '{s}' matched regex but didn't return int: {result}")
        except (ValueError, TypeError) as e:
            failures.append(f"Input '{s}' matched regex but raised: {e}")
    else:
        try:
            result = converter.to_python(s)
            failures.append(f"Input '{s}' didn't match regex but to_python succeeded: {result}")
        except (ValueError, TypeError):
            pass  # Expected behavior

# Run the property test
try:
    test_int_converter_contract_to_python_validates_regex()
    print(f"Checked {len(test_cases_checked)} test cases")
    if failures:
        print(f"Found {len(failures)} failures:")
        for i, failure in enumerate(failures[:10]):  # Show first 10 failures
            print(f"  {i+1}. {failure}")
    else:
        print("No failures found!")
except Exception as e:
    print(f"Test execution failed: {e}")
    if failures:
        print(f"Failures before error:")
        for i, failure in enumerate(failures[:10]):
            print(f"  {i+1}. {failure}")

# Test 3: Other edge cases
print("\n" + "=" * 50)
print("Test 3: Edge Cases")
print("=" * 50)

test_inputs = [
    '-1',
    '-100',
    '0',
    '1',
    '100',
    '1.5',
    'abc',
    '1a',
    'a1',
    '',
    ' 5',
    '5 ',
    '+5',
    '--5',
    '00123',
]

for test_input in test_inputs:
    matches_regex = bool(regex.match(test_input))
    try:
        result = converter.to_python(test_input)
        success = True
        exception = None
    except Exception as e:
        success = False
        exception = e
        result = None

    print(f"Input: '{test_input:8}' | Regex match: {matches_regex:5} | to_python success: {success:5} | Result: {result if success else f'{type(exception).__name__}'}")

# Test 4: Verify the actual URL routing behavior
print("\n" + "=" * 50)
print("Test 4: URL Routing Context")
print("=" * 50)

# Check how Django actually uses these converters in URL patterns
print("In actual Django URL routing:")
print("1. The regex is used FIRST to determine if a URL segment matches")
print("2. Only if regex matches, to_python() is called to convert the value")
print("3. If to_python() raises ValueError, the match fails (as seen in resolvers.py:332-334)")
print("\nFor IntConverter with regex '[0-9]+':")
print("  - URL '/articles/123/' -> regex matches '123' -> to_python('123') -> 123")
print("  - URL '/articles/-5/' -> regex doesn't match '-5' -> to_python() never called")
print("  - Direct call to_python('-5') -> works but bypasses regex validation")