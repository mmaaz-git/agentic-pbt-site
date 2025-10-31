import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper

seq = [10, 20]
results = list(looper(seq))

print("Testing type inconsistency between odd and even properties:")
print("=" * 60)

for i, (loop, item) in enumerate(results):
    print(f"\nItem at index {i} (value: {item}):")
    print(f"  loop.odd: {loop.odd!r} (type: {type(loop.odd).__name__})")
    print(f"  loop.even: {loop.even!r} (type: {type(loop.even).__name__})")

    # Also check other similar properties for comparison
    print(f"  loop.first: {loop.first!r} (type: {type(loop.first).__name__})")
    print(f"  loop.last: {loop.last!r} (type: {type(loop.last).__name__})")

print("\n" + "=" * 60)
print("BUG: The 'even' property returns int (0 or 1) instead of bool")
print("     while 'odd' property correctly returns bool (True or False)")