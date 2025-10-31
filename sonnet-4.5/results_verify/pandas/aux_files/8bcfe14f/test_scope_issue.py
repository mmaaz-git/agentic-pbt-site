"""Demonstrate why the scoping bug is problematic"""

# This simulates the structure found in pandas.core.indexes.api

def outer_function(indexes):
    """This simulates union_indexes"""

    def _find_common_index_dtype(inds):
        """
        This function has a bug - it declares parameter 'inds'
        but uses 'indexes' from outer scope
        """
        # BUG: Uses 'indexes' instead of 'inds'
        result = [item for item in indexes]
        return result

    # When called with 'indexes', it works by accident
    result1 = _find_common_index_dtype(indexes)
    print(f"Called with outer 'indexes': {result1}")

    # But if we wanted to call it with something else...
    other_data = ['x', 'y', 'z']
    result2 = _find_common_index_dtype(other_data)
    print(f"Called with different data: {result2}")
    print(f"Expected: ['x', 'y', 'z'], but got: {result2}")

    return result1 == result2

# Test it
test_data = ['a', 'b', 'c']
bug_present = outer_function(test_data)

if bug_present:
    print("\nBUG DEMONSTRATED: Function ignores its parameter and always uses outer scope variable!")
    print("This violates the principle of function locality and makes the code:")
    print("1. Confusing for maintainers")
    print("2. Error-prone if refactored")
    print("3. Misleading about its actual behavior")