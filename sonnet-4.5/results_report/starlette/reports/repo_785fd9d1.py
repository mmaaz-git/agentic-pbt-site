from starlette.convertors import FloatConvertor

convertor = FloatConvertor()

original = "0.000000000000000000001"
value = convertor.convert(original)
result = convertor.to_string(value)

print(f"Input:  '{original}'")
print(f"Float:  {value}")
print(f"Output: '{result}'")

# Check if round-trip works
roundtrip_value = convertor.convert(result)
print(f"Round-trip value: {roundtrip_value}")
print(f"Original float: {float(original)}")

# This will fail
assert result == original, f"Round-trip failed: '{original}' -> {value} -> '{result}'"