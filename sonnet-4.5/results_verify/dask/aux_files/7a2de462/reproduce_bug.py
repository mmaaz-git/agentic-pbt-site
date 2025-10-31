import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db

bag = db.from_sequence([1, 2, 3])

try:
    result = bag.join(42, lambda x: x)
    print("ERROR: Should have raised an exception")
except AttributeError as e:
    print(f"AttributeError caught: {e}")
    print(f"Expected TypeError but got AttributeError instead!")
except TypeError as e:
    print(f"TypeError (expected): {e}")