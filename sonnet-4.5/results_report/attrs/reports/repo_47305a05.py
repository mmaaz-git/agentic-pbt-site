import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from attr._cmp import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(f"Error message: {e}")