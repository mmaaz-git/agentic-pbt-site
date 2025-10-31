from pandas.io.json import ujson_dumps, ujson_loads
import json
import math

value = 1.7976931345e+308

stdlib_result = json.loads(json.dumps(value))
print(f"stdlib json:  {value} -> {stdlib_result} (finite: {math.isfinite(stdlib_result)})")

ujson_result = ujson_loads(ujson_dumps(value))
print(f"ujson:        {value} -> {ujson_result} (finite: {math.isfinite(ujson_result)})")

print(f"\nSerialized by ujson: {ujson_dumps(value)}")
print(f"Serialized by stdlib: {json.dumps(value)}")

assert stdlib_result == value, f"stdlib should preserve value: {stdlib_result} != {value}"
assert ujson_result != value, f"ujson fails to preserve value: {ujson_result} == {value}"

print("\nAlso testing with higher precision:")
result_high_precision = ujson_loads(ujson_dumps(value, double_precision=15))
print(f"ujson (precision=15): {value} -> {result_high_precision} (finite: {math.isfinite(result_high_precision)})")
assert result_high_precision == value, "High precision should preserve value"