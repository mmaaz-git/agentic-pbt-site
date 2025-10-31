import pandas as pd

# Create date ranges with different frequencies
start = '2025-01-01'

# Create daily range (D)
daily = pd.date_range(start, periods=10, freq='D')
print("Daily (D):", daily.to_list()[:5])

# Create business daily range (B)
business = pd.date_range(start, periods=10, freq='B')
print("Business (B):", business.to_list()[:5])

# Create custom business daily range (C)
custom = pd.date_range(start, periods=10, freq='C')
print("Custom (C):", custom.to_list()[:5])

# Test relationship: Business days are a subset of calendar days
print("\nBusiness days are weekdays, which are a subset of all days")
print("Business day includes Mon-Fri, excludes Sat-Sun")
print("Calendar day includes all days Mon-Sun")

# Understanding the relationship
print("\nConceptually:")
print("- If we have daily data (D), we CAN downsample to business days (B) by selecting weekdays only")
print("- If we have business day data (B), we CAN upsample to daily (D) by filling gaps")
print("- The same logic applies for custom business days (C)")