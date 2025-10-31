#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

test_input = '1.192092896e-07'
result = normalise_float_repr(test_input)

print(f"Input:    {test_input}")
print(f"Output:   {result}")
print(f"Expected: .0000001192092896")
print()
print(f"Input float value:  {float(test_input)}")
print(f"Output float value: {float(result)}")
print()
print(f"Are they equal? {float(test_input) == float(result)}")
print()

# Let's also trace through the function manually for understanding
print("Manual trace:")
str_value = test_input.lower().lstrip('0')
print(f"  str_value after initial processing: '{str_value}'")

str_value_pre_exp, exp_str = str_value.split('e')
exp = int(exp_str)
print(f"  str_value before exp split: '{str_value_pre_exp}'")
print(f"  exp after parsing: {exp}")

# Process the decimal point
num_int_digits = str_value_pre_exp.index('.')
str_value_no_dot = str_value_pre_exp[:num_int_digits] + str_value_pre_exp[num_int_digits + 1:]
print(f"  num_int_digits: {num_int_digits}")
print(f"  str_value without dot: '{str_value_no_dot}'")

exp_adjusted = exp + num_int_digits
print(f"  exp after adjustment: {exp_adjusted}")

# The problematic calculation
print(f"  str_value_no_dot[:exp_adjusted] = str_value_no_dot[:{exp_adjusted}] = '{str_value_no_dot[:exp_adjusted]}'")
print(f"  str_value_no_dot[exp_adjusted:] = str_value_no_dot[{exp_adjusted}:] = '{str_value_no_dot[exp_adjusted:]}'")

# Assertion to show the failure
assert float(test_input) == float(result), f"Values don't match! {float(test_input)} != {float(result)}"