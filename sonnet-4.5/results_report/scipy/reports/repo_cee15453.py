from scipy.io.arff._arffread import DateAttribute

# Test cases that expose the bug
test_patterns = [
    "date H",      # Single hour - should be 'h' but returns 'Y'
    "date m",      # Single minute - should be 'm' but returns 'Y'
    "date s",      # Single second - should be 's' but returns 'Y'
    "date HH",     # Double hour - correctly returns 'h'
    "date mm",     # Double minute - correctly returns 'm'
    "date ss",     # Double second - correctly returns 's'
    "date MM-dd",  # Month-day - correctly returns 'D' (bug masked)
]

print("Testing scipy.io.arff DateAttribute._get_date_format bug")
print("=" * 60)
print()
print("Bug: Line 276 contains 'elif \"yy\":' instead of 'elif \"yy\" in pattern:'")
print("This causes the condition to always evaluate to True.")
print()
print(f"Proof: bool('yy') = {bool('yy')} (always True!)")
print()
print("Test Results:")
print("-" * 60)

for pattern in test_patterns:
    try:
        result_fmt, result_unit = DateAttribute._get_date_format(pattern)
        # Determine expected unit based on pattern content
        if "H" in pattern or "HH" in pattern:
            expected = "h"
        elif "m" in pattern and "mm" not in pattern:  # single 'm'
            expected = "m"
        elif "mm" in pattern:
            expected = "m"
        elif "s" in pattern and "ss" not in pattern:  # single 's'
            expected = "s"
        elif "ss" in pattern:
            expected = "s"
        elif "dd" in pattern:
            expected = "D"
        elif "MM" in pattern:
            expected = "M"
        else:
            expected = "?"

        is_correct = result_unit == expected or (pattern in ["date MM-dd"] and result_unit == "D")
        status = "✓" if is_correct else "✗ BUG EXPOSED"

        print(f"Pattern: {pattern:<15} Result unit: {result_unit:<3} Expected: {expected:<3} {status}")
    except ValueError as e:
        print(f"Pattern: {pattern:<15} Error: {e}")

print()
print("Detailed Analysis for 'date H':")
print("-" * 60)
print("Execution flow:")
print("1. Pattern 'H' enters _get_date_format")
print("2. Line 273: 'yyyy' in 'H' -> False (skip)")
print("3. Line 276: 'yy' -> Always True (BUG!)")
print("4. Line 277: pattern.replace('yy', '%y') -> No change")
print("5. Line 278: datetime_unit = 'Y' (WRONG!)")
print("6. No HH/mm/ss checks match to overwrite")
print("7. Returns datetime_unit = 'Y' (should be 'h')")