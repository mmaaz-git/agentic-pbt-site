from pydantic.v1.types import ByteSize

# Test fractional bytes
bs = ByteSize.validate("0.5b")
print(f"ByteSize.validate('0.5b') = {bs}")
print(f"bs.to('b') = {bs.to('b')}")
print()

# Test multiple fractional values
test_values = ["0.1b", "0.9b", "0.5b", "0.9999b"]
for value in test_values:
    bs = ByteSize.validate(value)
    result = bs.to('b')
    print(f"Input: {value:8} -> ByteSize: {bs} -> Back to bytes: {result}")
print()

# Test larger units with fractions
test_kb = ["1.7kb", "2.5mb", "0.5kb"]
for value in test_kb:
    bs = ByteSize.validate(value)
    unit = value[-2:] if value.endswith("mb") else value[-2:]
    result = bs.to(unit)
    print(f"Input: {value:8} -> ByteSize: {bs:7} -> Back to {unit}: {result}")