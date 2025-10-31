from pydantic.v1.types import ByteSize

# Check inheritance
print(f"ByteSize inherits from: {ByteSize.__bases__}")
print(f"Is ByteSize a subclass of int? {issubclass(ByteSize, int)}")
print()

# Test int constraints
print("Testing int constraints:")
try:
    # Can't store a float in an int
    x = int(0.5)
    print(f"int(0.5) = {x}")
    print(f"Type of int(0.5): {type(x)}")
except Exception as e:
    print(f"Error: {e}")

print()

# Since ByteSize inherits from int, it must store integer values
bs = ByteSize(1234)
print(f"ByteSize(1234) = {bs}")
print(f"Type: {type(bs)}")
print(f"Is instance of int? {isinstance(bs, int)}")
print(f"Is instance of ByteSize? {isinstance(bs, ByteSize)}")

# Test arithmetic operations
print()
print("Arithmetic operations:")
print(f"bs + 100 = {bs + 100}, type: {type(bs + 100)}")
print(f"bs * 2 = {bs * 2}, type: {type(bs * 2)}")
print(f"bs / 2 = {bs / 2}, type: {type(bs / 2)}")  # Division produces float