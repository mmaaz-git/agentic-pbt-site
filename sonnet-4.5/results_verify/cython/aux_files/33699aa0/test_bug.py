import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper

seq = [10, 20]
results = list(looper(seq))

first_loop, _ = results[0]
second_loop, _ = results[1]

print("Testing first item (index 0):")
print(f"  odd: {first_loop.odd!r} (type: {type(first_loop.odd).__name__})")
print(f"  even: {first_loop.even!r} (type: {type(first_loop.even).__name__})")

print("\nTesting second item (index 1):")
print(f"  odd: {second_loop.odd!r} (type: {type(second_loop.odd).__name__})")
print(f"  even: {second_loop.even!r} (type: {type(second_loop.even).__name__})")

print("\nTesting other properties for comparison:")
print(f"  first_loop.first: {first_loop.first!r} (type: {type(first_loop.first).__name__})")
print(f"  first_loop.last: {first_loop.last!r} (type: {type(first_loop.last).__name__})")