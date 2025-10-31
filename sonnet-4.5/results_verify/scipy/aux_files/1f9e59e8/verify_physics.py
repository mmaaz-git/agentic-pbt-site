from scipy.constants import convert_temperature

# Check what absolute zero (0 K) is in different temperature scales
print("Absolute Zero conversions:")
print(f"0 K = {convert_temperature(0, 'Kelvin', 'Celsius')} C")
print(f"0 K = {convert_temperature(0, 'Kelvin', 'Fahrenheit')} F")
print(f"0 K = {convert_temperature(0, 'Kelvin', 'Rankine')} R")
print()

print("0 R conversions:")
print(f"0 R = {convert_temperature(0, 'Rankine', 'Kelvin')} K")
print(f"0 R = {convert_temperature(0, 'Rankine', 'Celsius')} C")
print(f"0 R = {convert_temperature(0, 'Rankine', 'Fahrenheit')} F")
print()

# Check if the function performs pure mathematical conversion
print("Mathematical conversion check:")
print("From the code: Kelvin to Celsius is: tempo - 273.15")
print(f"-10 K to Celsius: -10 - 273.15 = {-10 - 273.15} (matches: {convert_temperature(-10, 'Kelvin', 'Celsius')})")
print()

# Verify physics understanding
print("Physics context:")
print("- Absolute zero in Kelvin: 0 K")
print("- Absolute zero in Rankine: 0 R")
print("- Absolute zero in Celsius: -273.15 C")
print("- Absolute zero in Fahrenheit: -459.67 F")
print()
print("Temperatures below absolute zero are:")
print("- Theoretically impossible in classical thermodynamics")
print("- Only possible in certain quantum systems with inverted population (negative temperature)")
print("- Not what typical users expect from a temperature conversion function")