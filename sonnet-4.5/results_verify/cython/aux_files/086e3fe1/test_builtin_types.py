from Cython.Build.Inline import safe_type

# Test various types
test_values = [
    42,                        # int
    3.14,                     # float
    True,                     # bool
    [1, 2, 3],               # list
    (1, 2, 3),               # tuple
    {'a': 1},                # dict
    "hello",                 # str
    3+4j,                    # complex
]

print("Testing safe_type with builtin types (context=None):")
for val in test_values:
    try:
        result = safe_type(val)
        print(f"  {type(val).__name__:10s}: {result}")
    except Exception as e:
        print(f"  {type(val).__name__:10s}: ERROR - {e}")

# Now test with custom class
class MyClass:
    pass

print("\nTesting with custom class:")
obj = MyClass()
try:
    result = safe_type(obj)
    print(f"  MyClass: {result}")
except AttributeError as e:
    print(f"  MyClass: ERROR - {e}")