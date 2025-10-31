from pandas.io.json import ujson_dumps, ujson_loads
import json

val = 1.0000000000000002e+16
print(f"Original value: {val!r}")

ujson_result = ujson_dumps(val)
print(f"ujson_dumps: {ujson_result}")

recovered_ujson = ujson_loads(ujson_result)
print(f"ujson_loads: {recovered_ujson!r}")
print(f"Precision lost: {val != recovered_ujson}")
print(f"Difference: {val - recovered_ujson}")

stdlib_result = json.dumps(val)
print(f"\nstdlib json.dumps: {stdlib_result}")
recovered_stdlib = json.loads(stdlib_result)
print(f"stdlib json.loads: {recovered_stdlib!r}")
print(f"stdlib preserves: {val == recovered_stdlib}")