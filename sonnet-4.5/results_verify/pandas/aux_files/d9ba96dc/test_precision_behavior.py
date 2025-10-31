import pandas as pd
import numpy as np

def test_precision_effect():
    """Test how precision parameter affects the actual interval boundaries"""

    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84375]
    x = pd.Series(values)

    print("Testing how precision affects interval boundaries:")
    print("=" * 60)

    for precision in [1, 2, 3, 5, 10, 15]:
        result, bins = pd.cut(x, bins=2, retbins=True, precision=precision)
        categories = result.cat.categories

        print(f"\nPrecision = {precision}:")
        print(f"  Bins array: {bins}")
        print(f"  Categories: {categories}")

        # Check if values match
        match0 = categories[0].left == bins[0]
        match1 = categories[0].right == bins[1]
        match2 = categories[1].left == bins[1]
        match3 = categories[1].right == bins[2]

        all_match = match0 and match1 and match2 and match3

        print(f"  All boundaries match bins: {all_match}")
        if not all_match:
            print(f"    cat[0].left == bins[0]: {match0}")
            print(f"    cat[0].right == bins[1]: {match1}")
            print(f"    cat[1].left == bins[1]: {match2}")
            print(f"    cat[1].right == bins[2]: {match3}")

def test_string_representation():
    """Test that precision affects the string representation"""

    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84375]
    x = pd.Series(values)

    print("\n\nTesting string representation vs actual values:")
    print("=" * 60)

    for precision in [1, 3, 5]:
        result, bins = pd.cut(x, bins=2, retbins=True, precision=precision)
        categories = result.cat.categories

        print(f"\nPrecision = {precision}:")
        print(f"  String repr: {categories[0]}")
        print(f"  Actual left: {categories[0].left}")
        print(f"  Actual right: {categories[0].right}")
        print(f"  From bins: [{bins[0]}, {bins[1]}]")

if __name__ == "__main__":
    test_precision_effect()
    test_string_representation()