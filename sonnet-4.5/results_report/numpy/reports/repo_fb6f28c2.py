import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

import numpy.ma as ma

m1 = [False]
m2 = [False]
result = ma.mask_or(m1, m2)
print(f"Result: {result}")