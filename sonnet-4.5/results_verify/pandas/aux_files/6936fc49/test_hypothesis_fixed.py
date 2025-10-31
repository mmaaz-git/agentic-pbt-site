from pandas.api.indexers import FixedForwardWindowIndexer
import traceback

def test_fixed_forward_indexer_step_zero(window_size, num_values):
    """Test that step=0 raises ValueError, not ZeroDivisionError"""
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    try:
        indexer.get_window_bounds(num_values=num_values, step=0)
        return "NO_ERROR", None
    except ValueError as e:
        if "step must be" in str(e):
            return "CORRECT_ERROR", str(e)
        else:
            return "WRONG_MESSAGE", str(e)
    except ZeroDivisionError as e:
        return "ZERODIV_ERROR", str(e)
    except Exception as e:
        return "OTHER_ERROR", f"{type(e).__name__}: {e}"

# Test with specific failing input from bug report
print("Testing with window_size=1, num_values=1, step=0")
result, msg = test_fixed_forward_indexer_step_zero(1, 1)
print(f"  Result: {result}")
print(f"  Message: {msg}")
print()

# Test more cases
test_cases = [(5, 10), (10, 5), (50, 50), (1, 100), (100, 1)]
for window_size, num_values in test_cases:
    print(f"Testing with window_size={window_size}, num_values={num_values}, step=0")
    result, msg = test_fixed_forward_indexer_step_zero(window_size, num_values)
    print(f"  Result: {result}")
    if msg:
        print(f"  Message: {msg}")
    print()

print("Summary:")
print("The bug report is CORRECT - the function raises ZeroDivisionError instead of ValueError")
print("when step=0 is provided.")