from hypothesis import given, strategies as st, settings
from scipy.constants import convert_temperature

# First, run the hypothesis test
@given(st.floats(min_value=-1000, max_value=-0.01, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_negative_kelvin_accepted(negative_kelvin):
    """
    Bug: convert_temperature accepts negative Kelvin values.
    Negative Kelvin is physically impossible (absolute zero is 0 K).
    """
    result = convert_temperature(negative_kelvin, 'Kelvin', 'Celsius')
    assert negative_kelvin < 0
    print(f"Tested: {negative_kelvin} K -> {result} C")

# Test with a few values
print("Running hypothesis test with negative Kelvin values...")
test_negative_kelvin_accepted()
print("Hypothesis test passed - function accepts negative Kelvin\n")

# Now reproduce the specific examples
print("Reproducing specific examples from bug report:")
print("="*50)

print("Negative Kelvin accepted:")
result = convert_temperature(-10, 'Kelvin', 'Celsius')
print(f"  -10 K -> {result} C")

print("\nNegative Rankine accepted:")
result = convert_temperature(-10, 'Rankine', 'Fahrenheit')
print(f"  -10 R -> {result} F")

print("\nCelsius below absolute zero produces negative Kelvin:")
result = convert_temperature(-500, 'Celsius', 'Kelvin')
print(f"  -500 C -> {result} K (physically impossible!)")

print("\nAdditional test - extreme negative Kelvin:")
result = convert_temperature(-1000, 'Kelvin', 'Celsius')
print(f"  -1000 K -> {result} C")

# Verify the mathematical calculations
print("\n" + "="*50)
print("Verifying calculations manually:")
print("  -10 K to Celsius: -10 - 273.15 = -283.15 C ✓")
print("  -10 R to Fahrenheit: (-10 * 5/9 - 273.15) * 9/5 + 32 = -469.67 F ✓")
print("  -500 C to Kelvin: -500 + 273.15 = -226.85 K ✓")