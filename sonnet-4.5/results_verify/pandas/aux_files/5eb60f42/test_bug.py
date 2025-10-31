#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.io.clipboard as clipboard

# First test: try the hypothesis test
print("Test 1: Hypothesis test with timeout=0.0625")
clipboard.set_clipboard('no')
try:
    result = clipboard.waitForPaste(timeout=0.0625)
    print(f"  Result: Got '{result}' - should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    print(f"  Result: Correctly raised PyperclipTimeoutException: {e}")
except clipboard.PyperclipException as e:
    print(f"  Result: INCORRECTLY raised PyperclipException: {e}")
except Exception as e:
    print(f"  Result: Raised unexpected exception {type(e).__name__}: {e}")

# Second test: the reproduction case with 1 second timeout
print("\nTest 2: Reproduction with timeout=1.0")
clipboard.set_clipboard('no')
start_time = time.time()
try:
    result = clipboard.waitForPaste(timeout=1.0)
    print(f"  Result: Got '{result}' - should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start_time
    print(f"  Result: Correctly raised PyperclipTimeoutException after {elapsed:.3f}s: {e}")
except clipboard.PyperclipException as e:
    elapsed = time.time() - start_time
    print(f"  Result: INCORRECTLY raised PyperclipException after {elapsed:.3f}s: {e}")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"  Result: Raised unexpected exception {type(e).__name__} after {elapsed:.3f}s: {e}")

# Third test: waitForNewPaste
print("\nTest 3: waitForNewPaste with timeout=0.1")
clipboard.set_clipboard('no')
start_time = time.time()
try:
    result = clipboard.waitForNewPaste(timeout=0.1)
    print(f"  Result: Got '{result}' - should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start_time
    print(f"  Result: Correctly raised PyperclipTimeoutException after {elapsed:.3f}s: {e}")
except clipboard.PyperclipException as e:
    elapsed = time.time() - start_time
    print(f"  Result: INCORRECTLY raised PyperclipException after {elapsed:.3f}s: {e}")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"  Result: Raised unexpected exception {type(e).__name__} after {elapsed:.3f}s: {e}")

# Fourth test: Check what paste() does directly with no clipboard
print("\nTest 4: Direct paste() call with no clipboard")
clipboard.set_clipboard('no')
try:
    result = clipboard.paste()
    print(f"  Result: paste() returned: '{result}'")
except clipboard.PyperclipException as e:
    print(f"  Result: paste() raised PyperclipException: {e}")
except Exception as e:
    print(f"  Result: paste() raised {type(e).__name__}: {e}")