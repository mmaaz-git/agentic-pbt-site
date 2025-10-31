from pandas.io.json import ujson_dumps, ujson_loads

val = 1.0000000000000002e+16
print(f"Original value: {val!r}")

# Try with default precision
ujson_result_default = ujson_dumps(val)
print(f"\nDefault precision:")
print(f"  ujson_dumps: {ujson_result_default}")
recovered_default = ujson_loads(ujson_result_default)
print(f"  ujson_loads: {recovered_default!r}")
print(f"  Precision lost: {val != recovered_default}")

# Try with higher precision
for precision in [10, 15, 17, 20]:
    ujson_result = ujson_dumps(val, double_precision=precision)
    print(f"\nWith double_precision={precision}:")
    print(f"  ujson_dumps: {ujson_result}")
    recovered = ujson_loads(ujson_result)
    print(f"  ujson_loads: {recovered!r}")
    print(f"  Precision preserved: {val == recovered}")