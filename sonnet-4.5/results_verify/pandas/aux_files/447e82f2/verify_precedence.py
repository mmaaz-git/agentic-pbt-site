"""Verify the operator precedence issue"""

def show_evaluation(s):
    print(f"\nInput: {s!r}")
    print(f"  s[0] == '[': {s[0] == '['}")
    print(f"  s[-1] == ']': {s[-1] == ']'}")

    # Current buggy condition
    condition_buggy = not s[0] == "[" and s[-1] == "]"
    print(f"  Current condition (not s[0] == '[' and s[-1] == ']'): {condition_buggy}")
    print(f"    Evaluates as: (not (s[0] == '[')) and (s[-1] == ']')")
    print(f"    = {not (s[0] == '[')} and {s[-1] == ']'} = {condition_buggy}")

    # What the developer likely intended
    condition_intended = not (s[0] == "[" and s[-1] == "]")
    print(f"  Intended condition (not (s[0] == '[' and s[-1] == ']')): {condition_intended}")
    print(f"    = not ({s[0] == '['} and {s[-1] == ']'}) = {condition_intended}")

    # Alternative correct form using De Morgan's law
    condition_demorgan = s[0] != "[" or s[-1] != "]"
    print(f"  De Morgan's form (s[0] != '[' or s[-1] != ']'): {condition_demorgan}")

    print(f"\n  Result:")
    print(f"    Current code returns unchanged: {condition_buggy}")
    print(f"    Should return unchanged: {condition_intended}")
    print(f"    Match: {condition_buggy == condition_intended}")

# Test various cases
test_cases = ["abc", "[invalid", "test]", "[valid]", "{}"]

for s in test_cases:
    show_evaluation(s)