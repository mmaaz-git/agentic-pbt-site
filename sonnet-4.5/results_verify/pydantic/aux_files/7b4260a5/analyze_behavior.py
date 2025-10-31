import re
from pydantic.alias_generators import to_pascal

def analyze_to_pascal_step_by_step(input_str):
    """Analyze how to_pascal works step by step"""
    print(f"\nAnalyzing: '{input_str}'")

    # Step 1: title() transformation
    titled = input_str.title()
    print(f"  After .title(): '{titled}'")

    # Step 2: regex substitution
    pattern = r'([0-9A-Za-z])_(?=[0-9A-Z])'

    def replacement(m):
        result = m.group(1)
        print(f"    Regex match: '{m.group(0)}' -> '{result}'")
        return result

    result = re.sub(pattern, replacement, titled)
    print(f"  After regex: '{result}'")

    return result

# Test the problematic cases
test_cases = ['A_A', 'AA', 'a_b', 'AB']

for test in test_cases:
    result = analyze_to_pascal_step_by_step(test)
    print(f"  Final: to_pascal('{test}') = '{result}'")

    # Apply again
    print(f"  Applying again:")
    result2 = analyze_to_pascal_step_by_step(result)
    print(f"  Double application: to_pascal(to_pascal('{test}')) = '{result2}'")
    print(f"  Idempotent: {result == result2}")