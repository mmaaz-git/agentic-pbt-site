from pydantic.v1.types import ByteSize

# Test with 0.5 bytes
bs = ByteSize.validate("0.5b")
print(f"ByteSize object for '0.5b': {bs}")
print(f"Converting back to bytes: {bs.to('b')}")
print()

# Test with 1.7 kb
bs2 = ByteSize.validate("1.7kb")
print(f"ByteSize object for '1.7kb': {bs2}")
print(f"Converting back to kb: {bs2.to('kb')}")
print()

# Test with other fractional values
test_values = [("0.1b", "b"), ("0.9b", "b"), ("2.5mb", "mb"), ("3.3gb", "gb")]
for val, unit in test_values:
    bs = ByteSize.validate(val)
    print(f"Input: {val} -> ByteSize: {bs} -> Back to {unit}: {bs.to(unit)}{unit}")