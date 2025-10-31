from scipy import integrate

# Test case that should crash with IndexError
result = integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)
print(f"Result: {result}")