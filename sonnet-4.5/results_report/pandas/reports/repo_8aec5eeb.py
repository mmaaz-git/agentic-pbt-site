import pandas.errors as pe


class DummyClass:
    pass


try:
    pe.AbstractMethodError(DummyClass(), methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")
    print()
    print("Analysis:")
    print(f"  The error says 'methodtype must be one of invalid_type'")
    print(f"  But 'invalid_type' is the INVALID value we passed")
    print(f"  The valid values {{'method', 'classmethod', 'staticmethod', 'property'}} appear after 'got'")
    print()
    print("Expected format:")
    print("  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid_type instead.")