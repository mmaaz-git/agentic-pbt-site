#!/usr/bin/env python3
"""Understand why 1.1E-5 works but 6.103515625e-05 doesn't"""

def trace_execution(float_str):
    """Manually trace the execution like normalise_float_repr"""
    print(f"\nTracing: {float_str}")
    print("-" * 40)

    # Step 1: lower and lstrip
    str_value = float_str.lower().lstrip('0')
    print(f"After lower/lstrip: '{str_value}'")

    # Step 2: Handle exponent
    exp = 0
    if 'e' in str_value:
        mantissa, exp_str = str_value.split('e', 1)
        exp = int(exp_str)
        str_value = mantissa
        print(f"Mantissa: '{str_value}', Exp: {exp}")

    # Step 3: Handle decimal
    if '.' in str_value:
        num_int_digits = str_value.index('.')
        str_value_no_dot = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
        print(f"Decimal at position: {num_int_digits}")
        print(f"String without dot: '{str_value_no_dot}'")
    else:
        num_int_digits = len(str_value)
        str_value_no_dot = str_value
        print(f"No decimal, num_int_digits: {num_int_digits}")

    # Step 4: Adjust exp
    exp_adjusted = exp + num_int_digits
    print(f"Adjusted exp: {exp} + {num_int_digits} = {exp_adjusted}")

    # Step 5: Build result using negative indexing when exp_adjusted < 0
    print(f"\nBuilding result from '{str_value_no_dot}':")
    print(f"  Length of string: {len(str_value_no_dot)}")

    if exp_adjusted < 0:
        print(f"  exp_adjusted is negative ({exp_adjusted})")
        print(f"  str_value_no_dot[:{exp_adjusted}] = '{str_value_no_dot[:exp_adjusted]}'")
        print(f"  This takes the first {len(str_value_no_dot) + exp_adjusted} characters")
    else:
        print(f"  exp_adjusted is positive ({exp_adjusted})")
        print(f"  str_value_no_dot[:{exp_adjusted}] = '{str_value_no_dot[:exp_adjusted]}'")

    # Build the actual result string
    part1 = str_value_no_dot[:exp_adjusted]
    part2 = '0' * max(0, exp_adjusted - len(str_value_no_dot))
    part3 = '.'
    part4 = '0' * max(0, -exp_adjusted)
    part5 = str_value_no_dot[exp_adjusted:]

    print(f"\n  Parts:")
    print(f"    part1 (str_value_no_dot[:{exp_adjusted}]): '{part1}'")
    print(f"    part2 ('0' * {max(0, exp_adjusted - len(str_value_no_dot))}): '{part2}'")
    print(f"    part3 (decimal): '.'")
    print(f"    part4 ('0' * {max(0, -exp_adjusted)}): '{part4}'")
    print(f"    part5 (str_value_no_dot[{exp_adjusted}:]): '{part5}'")

    result = (part1 + part2 + part3 + part4 + part5).rstrip('0')
    if result == '.':
        result = '.0'

    print(f"\n  Final result: '{result}'")

    # Check correctness
    try:
        orig_val = float(float_str)
        result_val = float(result)
        print(f"\n  Original value: {orig_val}")
        print(f"  Result value:   {result_val}")
        print(f"  Correct: {abs(orig_val - result_val) < 1e-10}")
    except:
        print(f"  Cannot convert to float")


# Compare the working and failing cases
trace_execution('1.1E-5')
trace_execution('12.3E-5')
trace_execution('6.103515625e-05')

# Let's also check some edge cases
print("\n" + "="*60)
print("Understanding the pattern:")
print("="*60)

test_cases = [
    '1.0e-5',    # 1 digit after decimal
    '1.00e-5',   # 2 digits after decimal
    '1.000e-5',  # 3 digits after decimal
    '1.0000e-5', # 4 digits after decimal
    '1.00000e-5',# 5 digits after decimal
    '6.1e-5',    # Similar to bug case but fewer digits
    '6.10e-5',
    '6.103e-5',
    '6.1035e-5',
]

for tc in test_cases:
    str_value = tc.lower().lstrip('0')
    mantissa, exp_str = str_value.split('e')
    exp = int(exp_str)
    num_int_digits = mantissa.index('.')
    exp_adjusted = exp + num_int_digits
    mantissa_no_dot = mantissa.replace('.', '')

    from Cython.Utils import normalise_float_repr
    result = normalise_float_repr(tc)
    try:
        correct = abs(float(tc) - float(result)) < 1e-10
        status = "OK" if correct else "FAIL"
    except:
        status = "ERROR"

    print(f"{status:5} {tc:15} -> exp_adjusted={exp_adjusted:3}, mantissa='{mantissa_no_dot:10}', result='{result}'")