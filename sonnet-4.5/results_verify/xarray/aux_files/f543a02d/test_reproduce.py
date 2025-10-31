import threading
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages")
from xarray.backends.locks import CombinedLock

lock = threading.Lock()
combined = CombinedLock([lock])

print(f"lock.locked() = {lock.locked()}")
print(f"combined.locked() = {combined.locked()}")