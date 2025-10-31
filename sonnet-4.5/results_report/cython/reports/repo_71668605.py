import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utility.Dataclasses import field

# Create a field with kw_only=True
f = field(kw_only=True)

# Print the repr to show the inconsistency
print("repr(f) output:")
print(repr(f))

print("\nAccessing the actual attribute:")
print(f"f.kw_only = {f.kw_only}")

print("\nTrying to access f.kwonly (without underscore):")
try:
    print(f"f.kwonly = {f.kwonly}")
except AttributeError as e:
    print(f"AttributeError: {e}")