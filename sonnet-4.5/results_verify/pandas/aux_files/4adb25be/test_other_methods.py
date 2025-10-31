#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
from pandas.plotting._misc import _Options

print("Testing various dict methods that could remove default keys:")
print("="*60)

opts = _Options()
print(f"Initial state: {dict(opts)}")

# Test pop (confirmed issue)
print("\n1. Testing pop():")
opts1 = _Options()
try:
    result = opts1.pop("xaxis.compat")
    print(f"   pop succeeded (BUG): returned {result}, state: {dict(opts1)}")
except ValueError as e:
    print(f"   pop raised ValueError (correct): {e}")

# Test popitem
print("\n2. Testing popitem():")
opts2 = _Options()
try:
    result = opts2.popitem()
    print(f"   popitem succeeded: returned {result}, state: {dict(opts2)}")
    # Check if it was the default key
    if result[0] == "xaxis.compat":
        print("   WARNING: popitem removed default key!")
except ValueError as e:
    print(f"   popitem raised ValueError: {e}")
except KeyError as e:
    print(f"   popitem raised KeyError: {e}")

# Test clear
print("\n3. Testing clear():")
opts3 = _Options()
try:
    opts3.clear()
    print(f"   clear succeeded: state: {dict(opts3)}")
    if "xaxis.compat" not in opts3:
        print("   WARNING: clear removed default key!")
except ValueError as e:
    print(f"   clear raised ValueError: {e}")

# Test setdefault with None
print("\n4. Testing setdefault() to override:")
opts4 = _Options()
try:
    result = opts4.setdefault("xaxis.compat", True)
    print(f"   setdefault returned: {result}, state: {dict(opts4)}")
except ValueError as e:
    print(f"   setdefault raised ValueError: {e}")

# Test update to potentially remove keys
print("\n5. Testing update() with empty dict:")
opts5 = _Options()
try:
    opts5.update({})
    print(f"   update succeeded: state: {dict(opts5)}")
except ValueError as e:
    print(f"   update raised ValueError: {e}")

# Test __delitem__ for comparison (already confirmed working)
print("\n6. Testing del (for comparison):")
opts6 = _Options()
try:
    del opts6["xaxis.compat"]
    print(f"   del succeeded (unexpected): state: {dict(opts6)}")
except ValueError as e:
    print(f"   del raised ValueError (correct): {e}")