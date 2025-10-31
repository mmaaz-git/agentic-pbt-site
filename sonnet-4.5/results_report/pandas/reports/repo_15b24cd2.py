import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import build_hex_version

# Test version collisions with high serial numbers
alpha80 = build_hex_version("0.0a80")
final = build_hex_version("0.0")
print(f"0.0a80: {alpha80}")
print(f"0.0:    {final}")
print(f"Equal: {alpha80 == final}")
print()

beta64 = build_hex_version("0.0b64")
print(f"0.0b64: {beta64}")
print(f"Equal to final: {beta64 == final}")
print()

rc48 = build_hex_version("0.0rc48")
print(f"0.0rc48: {rc48}")
print(f"Equal to final: {rc48 == final}")
print()

# Show the hex values as integers for comparison
print("Integer values:")
print(f"0.0a80: {int(alpha80, 16)}")
print(f"0.0:    {int(final, 16)}")
print(f"0.0b64: {int(beta64, 16)}")
print(f"0.0rc48: {int(rc48, 16)}")