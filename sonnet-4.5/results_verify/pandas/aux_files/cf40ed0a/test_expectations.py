#!/usr/bin/env python3
"""Test what a user would reasonably expect from qcut"""

import pandas as pd
import numpy as np

print("User Expectation Test")
print("=" * 60)

print("\nScenario: A teacher wants to grade 20 students into 2 groups (top 50%, bottom 50%)")
print("Scores: 18 students scored 75%, 1 scored 90%, 1 scored 60%")

scores = [75] * 18 + [90, 60]
print(f"Scores: {scores}")

print("\nExpected behavior:")
print("- Bottom 50% (10 students): Would include the student with 60% and 9 with 75%")
print("- Top 50% (10 students): Would include 9 students with 75% and 1 with 90%")

print("\nWhat qcut actually does:")
result = pd.qcut(scores, 2, duplicates='drop')
print(f"Bins: {result.value_counts()}")

print("\n" + "=" * 60)
print("The problem: qcut uses the mathematical median, which doesn't split")
print("duplicates. This violates user expectations for 'equal-sized buckets'.")

print("\n" + "=" * 60)
print("Testing if this is documented behavior:")

# Check what happens with perfectly splittable data
print("\nTest with no duplicates at boundary:")
data_good = list(range(20))
result_good = pd.qcut(data_good, 2)
print(f"Data: range(20)")
print(f"Result: {result_good.value_counts()}")

print("\nTest with duplicates but still splittable:")
data_split = [1]*5 + [2]*5 + [3]*5 + [4]*5
result_split = pd.qcut(data_split, 2, duplicates='drop')
print(f"Data: [1]*5 + [2]*5 + [3]*5 + [4]*5")
print(f"Result: {result_split.value_counts()}")

print("\n" + "=" * 60)
print("Conclusion: When duplicates don't fall on quantile boundaries,")
print("qcut produces equal bins. But when they do, it fails badly.")
print("This is a legitimate issue with the algorithm.")