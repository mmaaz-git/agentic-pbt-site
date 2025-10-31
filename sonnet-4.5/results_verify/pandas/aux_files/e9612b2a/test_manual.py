from pandas.io.json import ujson_dumps, ujson_loads

# Test the specific value mentioned in the bug report (-2^64)
value = -18_446_744_073_709_551_616
print(f"Testing value: {value} (-2^64)")

serialized = ujson_dumps(value)
deserialized = ujson_loads(serialized)

print(f"Original:     {value}")
print(f"Serialized:   {serialized}")
print(f"Deserialized: {deserialized}")
print(f"Match:        {deserialized == value}")

# Also test with dictionary as mentioned in the bug
print("\nTesting with dictionary:")
d = {'0': -18446744073709551616}
print(f"Original dict: {d}")
serialized_dict = ujson_dumps(d)
print(f"Serialized: {serialized_dict}")
deserialized_dict = ujson_loads(serialized_dict)
print(f"Deserialized: {deserialized_dict}")
print(f"Match: {deserialized_dict == d}")

# Test a range of values around -2^64
print("\nTesting values around -2^64:")
test_values = [
    -18_446_744_073_709_551_615,  # -2^64 + 1
    -18_446_744_073_709_551_616,  # -2^64
    -18_446_744_073_709_551_617,  # -2^64 - 1
    -9_223_372_036_854_775_808,   # -2^63
    -9_223_372_036_854_775_809,   # -2^63 - 1
]

for val in test_values:
    try:
        s = ujson_dumps(val)
        d = ujson_loads(s)
        print(f"Value {val}: serialized={s}, deserialized={d}, match={d==val}")
    except Exception as e:
        print(f"Value {val}: ERROR - {e}")