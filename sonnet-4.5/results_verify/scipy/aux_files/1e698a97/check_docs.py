from scipy.spatial.distance import dice
import inspect

print("Dice function documentation:")
print("=" * 60)
print(dice.__doc__)
print("=" * 60)

# Also check the source code location
print(f"\nSource file: {inspect.getfile(dice)}")
print(f"Line number: {inspect.findsource(dice)[1] + 1}")