import numpy as np
import warnings

# Let's trace what happens when bmat is called with string and gdict only

print("When bmat('A', gdict={'A': matrix}) is called:")
print("1. bmat checks if obj is a string - YES")
print("2. gdict is not None, so it goes to else branch at line 1103-1105")
print("3. glob_dict = gdict (the provided dict)")
print("4. loc_dict = ldict (which is None since not provided)")
print("5. Calls _from_string('A', glob_dict={'A': matrix}, loc_dict=None)")
print("\nIn _from_string:")
print("6. At line 1030: thismat = ldict[col] where ldict=None")
print("7. This raises TypeError: 'NoneType' object is not subscriptable")
print("8. TypeError is NOT caught by the except KeyError handler")
print("9. Exception propagates up, function fails")

print("\n" + "="*50)
print("Testing the documentation example:")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)

    # These work because variables are in local scope
    A = np.asmatrix('1 1; 1 1')
    B = np.asmatrix('2 2; 2 2')
    C = np.asmatrix('3 4; 5 6')
    D = np.asmatrix('7 8; 9 0')

    # This works because A,B,C,D are in the calling frame
    result = np.bmat('A,B; C,D')
    print("np.bmat('A,B; C,D') works when variables are in local scope")
    print(f"Result:\n{result}")

print("\n" + "="*50)
print("What the documentation says about gdict:")
print("- 'A dictionary that replaces global operands in current frame.'")
print("- 'Ignored if `obj` is not a string.'")
print("\nWhat it doesn't say:")
print("- That gdict only works if ldict is also provided")
print("- That passing gdict without ldict will cause a TypeError")