import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

print("numpy.char.join documentation:")
print("=" * 50)
print(char.join.__doc__)
print("\n" + "=" * 50)

# Also check if there's help available
help(char.join)