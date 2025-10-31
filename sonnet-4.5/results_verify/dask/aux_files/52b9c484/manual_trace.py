#!/usr/bin/env python3
"""Manual trace through the _aggregate_columns function to understand the bug."""

def _aggregate_columns_traced(cols, agg_cols):
    """Traced version of the function to understand the infinite loop."""
    print(f"Called with cols={cols}, agg_cols={agg_cols}")

    combine = []
    i = 0
    iteration = 0
    max_iterations = 10  # Safety limit

    while True:
        iteration += 1
        if iteration > max_iterations:
            print(f"Stopping after {max_iterations} iterations to prevent infinite loop")
            break

        print(f"\nIteration {iteration}: i={i}")
        inner = []
        combine.append(inner)
        print(f"  Created empty inner list, combine now has {len(combine)} elements")

        try:
            print(f"  Looping through cols (length={len(cols)})...")
            for j, col in enumerate(cols):
                print(f"    Processing col[{j}][{i}] = {col[i] if i < len(col) else 'IndexError'}")
                inner.append(col[i])
            print(f"  Inner list after loop: {inner}")
        except IndexError as e:
            print(f"  IndexError caught: {e}")
            combine.pop()
            print(f"  Popped last element, combine now has {len(combine)} elements")
            break

        print(f"  No IndexError raised, incrementing i to {i+1}")
        i += 1

    print(f"\nFinal combine: {combine}")
    # Would normally do: return [_agg_dicts(c, agg_cols) for c in combine]
    return combine

# Test with empty list
print("=" * 60)
print("TEST 1: Empty list (bug case)")
print("=" * 60)
result = _aggregate_columns_traced([], {})
print(f"Result: {result}")

# Test with non-empty list
print("\n" + "=" * 60)
print("TEST 2: Non-empty list [[1,2], [3,4]]")
print("=" * 60)
result = _aggregate_columns_traced([[1,2], [3,4]], {})
print(f"Result: {result}")