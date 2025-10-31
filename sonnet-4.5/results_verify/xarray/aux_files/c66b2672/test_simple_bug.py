import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.compat.array_api_compat import result_type, is_weak_scalar_type, _future_array_api_result_type

print("Testing simple string case:")
print(f"is_weak_scalar_type('test'): {is_weak_scalar_type('test')}")

try:
    result = result_type("test", xp=np)
    print(f"result_type('test', xp=np): {result}")
except Exception as e:
    print(f"result_type('test', xp=np) raised: {type(e).__name__}: {e}")

print("\nTesting bytes case:")
print(f"is_weak_scalar_type(b'test'): {is_weak_scalar_type(b'test')}")

try:
    result = result_type(b"test", xp=np)
    print(f"result_type(b'test', xp=np): {result}")
except Exception as e:
    print(f"result_type(b'test', xp=np) raised: {type(e).__name__}: {e}")

print("\nTesting _future_array_api_result_type directly:")
try:
    result = _future_array_api_result_type("test", xp=np)
    print(f"_future_array_api_result_type('test', xp=np): {result}")
except Exception as e:
    print(f"_future_array_api_result_type('test', xp=np) raised: {type(e).__name__}: {e}")

print("\nTesting numpy's result_type directly:")
try:
    result = np.result_type("test")
    print(f"np.result_type('test'): {result}")
except Exception as e:
    print(f"np.result_type('test') raised: {type(e).__name__}: {e}")