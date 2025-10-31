import ctypes
import platform

print(f"Platform: {platform.system()}")
print(f"Checking if ctypes.WinError exists...")

if hasattr(ctypes, 'WinError'):
    print("ctypes.WinError exists!")
    try:
        result = ctypes.WinError()
        print(f"ctypes.WinError() returned: {result}")
    except Exception as e:
        print(f"ctypes.WinError() raised: {e}")
else:
    print("ctypes.WinError does NOT exist on this platform")

print("\nAll ctypes attributes containing 'Win':")
win_attrs = [attr for attr in dir(ctypes) if 'Win' in attr]
print(win_attrs if win_attrs else "None found")