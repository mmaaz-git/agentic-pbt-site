import attr

# Test how fields() handles non-class inputs
print("Testing attr.fields(42):")
try:
    result = attr.fields(42)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

print("\nTesting attr.fields('not a class'):")
try:
    result = attr.fields("not a class")
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

print("\nTesting attr.fields(None):")
try:
    result = attr.fields(None)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Test fields_dict() too
print("\n\nTesting attr.fields_dict(42):")
try:
    result = attr.fields_dict(42)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

print("\nTesting attr.fields_dict('not a class'):")
try:
    result = attr.fields_dict("not a class")
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Compare with has()
print("\n\nComparison with attr.has():")
print("attr.has(42):", attr.has(42))
print("attr.has('not a class'):", attr.has("not a class"))
print("attr.has(None):", attr.has(None))