#!/usr/bin/env python3
"""Minimal reproduction of the scipy.datasets._download_all main() crash when pooch is not installed."""

import sys
import os

# Ensure we're in a clean state without pooch
pooch_backup = sys.modules.get('pooch')
try:
    # Remove pooch from sys.modules to simulate it not being installed
    sys.modules['pooch'] = None

    # Now try to use the scipy.datasets._download_all.main() function
    from scipy.datasets._download_all import main
    import argparse

    print("Testing scipy.datasets._download_all.main() without pooch installed...")
    print("-" * 70)

    try:
        # This should fail gracefully with a clear ImportError message
        # about needing to install pooch, but instead it crashes with AttributeError
        main()
        print("ERROR: main() succeeded when it should have failed!")
    except AttributeError as e:
        print(f"AttributeError raised (WRONG ERROR TYPE): {e}")
        print(f"Full error type: {type(e).__name__}")
    except ImportError as e:
        print(f"ImportError raised (CORRECT): {e}")
        print(f"Full error type: {type(e).__name__}")
    except SystemExit as e:
        # argparse might exit if arguments are wrong
        print(f"SystemExit raised: {e}")
    except Exception as e:
        print(f"Unexpected error type {type(e).__name__}: {e}")

    print("-" * 70)
    print("\nEXPECTED: ImportError with message about installing pooch")
    print("ACTUAL: AttributeError: 'NoneType' object has no attribute 'os_cache'")

finally:
    # Restore original pooch module if it existed
    if pooch_backup:
        sys.modules['pooch'] = pooch_backup
    elif 'pooch' in sys.modules:
        del sys.modules['pooch']