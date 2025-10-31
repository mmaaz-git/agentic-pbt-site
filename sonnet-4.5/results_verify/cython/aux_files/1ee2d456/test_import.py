import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Tempita._tempita as tempita

print(f"Module file: {tempita.__file__}")
print(f"Is .so: {tempita.__file__.endswith('.so')}")