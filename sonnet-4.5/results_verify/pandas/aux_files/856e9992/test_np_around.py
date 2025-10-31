import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np

# Test np.around with large digits
test_value = 5e-324

for digits in [10, 50, 100, 200, 300, 320, 325, 326]:
    result = np.around(test_value, digits)
    print(f"np.around({test_value}, {digits}) = {result}")