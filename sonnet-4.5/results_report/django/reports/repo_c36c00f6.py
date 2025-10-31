from django.db.models.expressions import F
from django.db.models.functions import NthValue

# Test with nth=0
try:
    NthValue(F('field'), nth=0)
except ValueError as e:
    print(f"Test with nth=0:")
    print(f"Error message: {str(e)}")
    print()

# Test with nth=-1
try:
    NthValue(F('field'), nth=-1)
except ValueError as e:
    print(f"Test with nth=-1:")
    print(f"Error message: {str(e)}")
    print()

# Test with nth=None
try:
    NthValue(F('field'), nth=None)
except ValueError as e:
    print(f"Test with nth=None:")
    print(f"Error message: {str(e)}")
    print()

# For comparison, check the LagLeadFunction error message
from django.db.models.functions import Lag

try:
    Lag(F('field'), offset=0)
except ValueError as e:
    print(f"Comparison - Lag function with offset=0:")
    print(f"Error message: {str(e)}")
    print()

print("\nAnalysis:")
print("The NthValue error message contains 'as for nth' which is grammatically incorrect.")
print("It should be either 'for nth' or 'as the nth parameter'.")
print("Note how the Lag function correctly uses 'for the offset'.")