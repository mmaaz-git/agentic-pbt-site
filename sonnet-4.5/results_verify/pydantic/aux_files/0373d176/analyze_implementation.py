#!/usr/bin/env python3
"""Analyze the implementation to understand the behavior."""

import re

def analyze_to_snake_step_by_step(input_str):
    """Step-by-step analysis of to_snake implementation."""
    print(f"\nAnalyzing: '{input_str}'")
    print("=" * 50)

    current = input_str

    # Step 1: Handle the sequence of uppercase letters followed by a lowercase letter
    pattern1 = r'([A-Z]+)([A-Z][a-z])'
    result1 = re.sub(pattern1, lambda m: f'{m.group(1)}_{m.group(2)}', current)
    print(f"Step 1 - Pattern: {pattern1}")
    print(f"  '{current}' -> '{result1}'")
    current = result1

    # Step 2: Insert an underscore between a lowercase letter and an uppercase letter
    pattern2 = r'([a-z])([A-Z])'
    result2 = re.sub(pattern2, lambda m: f'{m.group(1)}_{m.group(2)}', current)
    print(f"Step 2 - Pattern: {pattern2}")
    print(f"  '{current}' -> '{result2}'")
    current = result2

    # Step 3: Insert an underscore between a digit and an uppercase letter
    pattern3 = r'([0-9])([A-Z])'
    result3 = re.sub(pattern3, lambda m: f'{m.group(1)}_{m.group(2)}', current)
    print(f"Step 3 - Pattern: {pattern3}")
    print(f"  '{current}' -> '{result3}'")
    current = result3

    # Step 4: Insert an underscore between a lowercase letter and a digit
    pattern4 = r'([a-z])([0-9])'
    result4 = re.sub(pattern4, lambda m: f'{m.group(1)}_{m.group(2)}', current)
    print(f"Step 4 - Pattern: {pattern4}")
    print(f"  '{current}' -> '{result4}'")
    current = result4

    # Step 5: Replace hyphens with underscores to handle kebab-case
    result5 = current.replace('-', '_')
    print(f"Step 5 - Replace hyphens:")
    print(f"  '{current}' -> '{result5}'")
    current = result5

    # Step 6: Convert to lowercase
    final = current.lower()
    print(f"Step 6 - Convert to lowercase:")
    print(f"  '{current}' -> '{final}'")

    return final

# Test the problematic cases
test_cases = ['A0', 'a0', 'AB0', 'Ab0']

for test in test_cases:
    result1 = analyze_to_snake_step_by_step(test)
    print(f"\nFirst application result: '{result1}'")

    result2 = analyze_to_snake_step_by_step(result1)
    print(f"Second application result: '{result2}'")

    print(f"Idempotent? {result1 == result2}")