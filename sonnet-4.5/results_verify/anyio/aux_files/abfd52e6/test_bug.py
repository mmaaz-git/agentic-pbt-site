#!/usr/bin/env python3
"""Test case to reproduce the anyio.sleep_until NaN bug"""

import math
import anyio
import time

# First test - simple reproduction
async def test_nan_simple():
    print("\n=== Test 1: Simple NaN test ===")
    print("Calling sleep_until(nan) with 1-second timeout...")
    start_time = time.time()
    try:
        with anyio.fail_after(1.0):
            await anyio.sleep_until(float('nan'))
        elapsed = time.time() - start_time
        print(f"Completed without timeout in {elapsed:.3f}s (unexpected!)")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"Timed out after {elapsed:.3f}s - sleep_until(nan) hung indefinitely")

# Test Python's max behavior with NaN
def test_max_nan():
    print("\n=== Test 2: Python max() behavior with NaN ===")
    print(f"max(float('nan'), 0) = {max(float('nan'), 0)}")
    print(f"max(0, float('nan')) = {max(0, float('nan'))}")
    nan_val = float('nan')
    print(f"math.isnan(max(nan, 0)) = {math.isnan(max(nan_val, 0))}")

# Test regular sleep with NaN
async def test_sleep_nan():
    print("\n=== Test 3: Direct sleep with NaN ===")
    print("Testing anyio.sleep(nan) with 1-second timeout...")
    start_time = time.time()
    try:
        with anyio.fail_after(1.0):
            await anyio.sleep(float('nan'))
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.3f}s")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"Timed out after {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Exception raised in {elapsed:.3f}s: {type(e).__name__}: {e}")

# Test other special float values
async def test_special_values():
    print("\n=== Test 4: Other special float values ===")

    # Test positive infinity
    print("Testing sleep_until(inf)...")
    start_time = time.time()
    try:
        with anyio.fail_after(0.5):
            await anyio.sleep_until(float('inf'))
        elapsed = time.time() - start_time
        print(f"  Infinity: completed in {elapsed:.3f}s")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"  Infinity: timed out after {elapsed:.3f}s (expected)")

    # Test negative infinity
    print("Testing sleep_until(-inf)...")
    start_time = time.time()
    try:
        with anyio.fail_after(0.5):
            await anyio.sleep_until(float('-inf'))
        elapsed = time.time() - start_time
        print(f"  Negative Infinity: completed in {elapsed:.3f}s (expected - past time)")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"  Negative Infinity: timed out after {elapsed:.3f}s")

# Test the calculation logic manually
async def test_calculation():
    print("\n=== Test 5: Manual calculation test ===")
    now = anyio.current_time()
    deadline = float('nan')
    delay = deadline - now
    print(f"current_time() = {now}")
    print(f"deadline = {deadline}")
    print(f"deadline - now = {delay}")
    print(f"math.isnan(deadline - now) = {math.isnan(delay)}")
    print(f"max(deadline - now, 0) = {max(delay, 0)}")
    print(f"math.isnan(max(deadline - now, 0)) = {math.isnan(max(delay, 0))}")

async def main():
    print("Starting anyio.sleep_until NaN bug reproduction tests...")
    print("=" * 60)

    # Run all tests
    test_max_nan()  # This is synchronous
    await test_calculation()
    await test_sleep_nan()
    await test_nan_simple()
    await test_special_values()

    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    anyio.run(main)