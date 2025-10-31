#!/usr/bin/env python3
"""Test asyncio.sleep behavior with NaN"""

import asyncio
import time
import math

async def test_asyncio_sleep_nan():
    print("Testing asyncio.sleep(nan)...")
    try:
        await asyncio.sleep(float('nan'))
        print("  Completed successfully (no error)")
    except ValueError as e:
        print(f"  ValueError raised: {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")

async def test_asyncio_sleep_inf():
    print("Testing asyncio.sleep(inf) with timeout...")
    try:
        await asyncio.wait_for(asyncio.sleep(float('inf')), timeout=0.5)
        print("  Completed (unexpected)")
    except asyncio.TimeoutError:
        print("  Timed out as expected")

# Test time.sleep with NaN
def test_time_sleep_nan():
    print("Testing time.sleep(nan)...")
    try:
        time.sleep(float('nan'))
        print("  Completed successfully (no error)")
    except ValueError as e:
        print(f"  ValueError raised: {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")

async def main():
    print("Python asyncio.sleep NaN behavior test")
    print("=" * 50)

    test_time_sleep_nan()
    await test_asyncio_sleep_nan()
    await test_asyncio_sleep_inf()

    print("\nConclusion: In Python 3.13, asyncio.sleep(nan) raises ValueError")

if __name__ == "__main__":
    asyncio.run(main())