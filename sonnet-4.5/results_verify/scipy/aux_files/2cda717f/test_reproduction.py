import sys

pooch_backup = sys.modules.get('pooch')
try:
    sys.modules['pooch'] = None

    from scipy.datasets._download_all import main
    import argparse

    try:
        main()
    except AttributeError as e:
        print(f"Error: {e}")
        print("Expected: ImportError with clear message about missing pooch")
        print("Got: AttributeError about NoneType")
finally:
    if pooch_backup:
        sys.modules['pooch'] = pooch_backup