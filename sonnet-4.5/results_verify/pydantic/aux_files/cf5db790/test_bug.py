"""Testing the reported bug in pydantic.deprecated.decorator.to_pascal"""

# First, let's test the basic reproduction case
from pydantic.deprecated.decorator import to_pascal

print("Testing basic reproduction case:")
print(f"to_pascal('a_a') = {to_pascal('a_a')!r}")
print(f"to_pascal('AA') = {to_pascal('AA')!r}")
print(f"to_pascal(to_pascal('a_a')) = {to_pascal(to_pascal('a_a'))!r}")

# Test the assertions from the bug report
try:
    assert to_pascal('a_a') == 'AA'
    print("✓ to_pascal('a_a') == 'AA'")
except AssertionError:
    print("✗ to_pascal('a_a') != 'AA'")

try:
    assert to_pascal('AA') == 'Aa'
    print("✓ to_pascal('AA') == 'Aa'")
except AssertionError:
    print("✗ to_pascal('AA') != 'Aa'")

try:
    assert to_pascal('a_a') != to_pascal(to_pascal('a_a'))
    print("✓ to_pascal('a_a') != to_pascal(to_pascal('a_a'))")
except AssertionError:
    print("✗ to_pascal('a_a') == to_pascal(to_pascal('a_a'))")