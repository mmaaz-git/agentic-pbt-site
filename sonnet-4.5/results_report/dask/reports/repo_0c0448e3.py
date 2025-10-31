import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db

# Create a simple dask bag
bag = db.from_sequence([1, 2, 3])

# Try to join with an invalid type (integer)
try:
    result = bag.join(42, lambda x: x)
    print("No error raised - this should not happen!")
except AttributeError as e:
    print(f"AttributeError caught (unexpected): {e}")
except TypeError as e:
    print(f"TypeError caught (expected): {e}")