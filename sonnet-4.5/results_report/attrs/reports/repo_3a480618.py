import attr

# Test case 1: Integer input
result = attr.has(42)
print(f"attr.has(42) = {result}")

# Test case 2: String input
result = attr.has("not a class")
print(f"attr.has('not a class') = {result}")

# Test case 3: None input
result = attr.has(None)
print(f"attr.has(None) = {result}")

# Test case 4: List input
result = attr.has([1, 2, 3])
print(f"attr.has([1, 2, 3]) = {result}")

# Test case 5: Dictionary input
result = attr.has({"key": "value"})
print(f"attr.has({{'key': 'value'}}) = {result}")

# Test case 6: Float input
result = attr.has(3.14)
print(f"attr.has(3.14) = {result}")

print("\nAll of the above should have raised TypeError according to documentation.")
print("Instead, they all returned False.")