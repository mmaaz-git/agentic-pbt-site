import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import time

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir, cache_size=500)

    old_file = os.path.join(tmpdir, "old.txt")
    with open(old_file, 'w') as f:
        f.write('x' * 200)

    time.sleep(0.1)

    new_file = os.path.join(tmpdir, "new.txt")
    with open(new_file, 'w') as f:
        f.write('x' * 200)

    time.sleep(0.1)

    newest_file = os.path.join(tmpdir, "newest.txt")
    with open(newest_file, 'w') as f:
        f.write('x' * 200)

    print(f"Files before cleanup: {sorted(os.listdir(tmpdir))}")

    cache.cleanup_cache(ratio=0.5)

    remaining = set(os.listdir(tmpdir))
    print(f"Files after cleanup: {sorted(remaining)}")

    if "old.txt" in remaining and "newest.txt" not in remaining:
        print("BUG CONFIRMED: Oldest file kept, newest file removed (opposite of LRU)")