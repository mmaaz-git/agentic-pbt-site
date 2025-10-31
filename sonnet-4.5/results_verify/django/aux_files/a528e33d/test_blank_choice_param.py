import django.db.models.fields as fields

# Test with the default BLANK_CHOICE_DASH
BLANK_CHOICE_DASH = [("", "---------")]
choices = [('a', 'A'), ('b', 'B')]

# BlankChoiceIterator receives blank_choice as a LIST containing a tuple
bci = fields.BlankChoiceIterator(choices, BLANK_CHOICE_DASH)
result = list(bci)

print("Testing with BLANK_CHOICE_DASH (list with one tuple):")
print(f"blank_choice parameter: {BLANK_CHOICE_DASH}")
print(f"Expected: {BLANK_CHOICE_DASH + choices}")
print(f"Actual: {result}")
print()

# The bug is that yield from unpacks the LIST, yielding the tuple
# But we want to yield each element of the list (which is just one tuple)

# Let's also test with a single tuple (not in a list)
blank_choice_tuple = ("", "Empty")
bci2 = fields.BlankChoiceIterator(choices, blank_choice_tuple)
result2 = list(bci2)

print("Testing with a single tuple (not in a list):")
print(f"blank_choice parameter: {blank_choice_tuple}")
print(f"Expected (if it was meant to yield the tuple): {[blank_choice_tuple] + choices}")
print(f"Actual: {result2}")