import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import key_split

# Test with invalid UTF-8 bytes
invalid_bytes = b'\x80'
try:
    result = key_split(invalid_bytes)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"Crashed with UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Crashed with other exception: {type(e).__name__}: {e}")