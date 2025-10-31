import pandas.errors as pe


class DummyClass:
    pass


obj = DummyClass()

# Test the bug
try:
    pe.AbstractMethodError(obj, methodtype='0')
except ValueError as e:
    print(f"Actual error message:   {e}")
    print(f"Expected error message: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got 0 instead.")

    # Check if bug exists
    error_msg = str(e)
    if "methodtype must be one of 0" in error_msg:
        print("\n[BUG CONFIRMED] The error message has swapped variables!")
        print("The invalid input '0' appears where valid types should be listed.")
    else:
        print("\n[NO BUG] The error message appears to be correct.")