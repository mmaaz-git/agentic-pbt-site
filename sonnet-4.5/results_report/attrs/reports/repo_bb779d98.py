from attrs import converters

print("Documented behavior:")
print(f"to_bool(1) = {converters.to_bool(1)}")
print(f"to_bool(0) = {converters.to_bool(0)}")
print(f"to_bool(True) = {converters.to_bool(True)}")
print(f"to_bool(False) = {converters.to_bool(False)}")
print(f'to_bool("1") = {converters.to_bool("1")}')
print(f'to_bool("0") = {converters.to_bool("0")}')

print("\nUndocumented behavior - accepts these floats:")
print(f"to_bool(1.0) = {converters.to_bool(1.0)}")
print(f"to_bool(0.0) = {converters.to_bool(0.0)}")

print("\nInconsistent behavior - rejects these floats:")
try:
    result = converters.to_bool(1.5)
    print(f"to_bool(1.5) = {result}")
except ValueError as e:
    print(f"to_bool(1.5) raises ValueError: {e}")

try:
    result = converters.to_bool(2.0)
    print(f"to_bool(2.0) = {result}")
except ValueError as e:
    print(f"to_bool(2.0) raises ValueError: {e}")

try:
    result = converters.to_bool(0.5)
    print(f"to_bool(0.5) = {result}")
except ValueError as e:
    print(f"to_bool(0.5) raises ValueError: {e}")

try:
    result = converters.to_bool(-1.0)
    print(f"to_bool(-1.0) = {result}")
except ValueError as e:
    print(f"to_bool(-1.0) raises ValueError: {e}")