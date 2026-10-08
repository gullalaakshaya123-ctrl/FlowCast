import pandas as pd


# --------------------------------------------------
# 1. Load calendar dataset
# --------------------------------------------------

calendar = pd.read_csv(
    "data/raw/calendar_events.csv"
)


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("========== CALENDAR DATASET ==========")

print("Rows:", len(calendar))
print("Columns:", len(calendar.columns))

print("\nColumns:")
print(list(calendar.columns))


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

print(
    calendar.isnull().sum()
)


# --------------------------------------------------
# 4. Duplicate records
# --------------------------------------------------

print("\n========== DUPLICATES ==========")

print(
    "Duplicate rows:",
    calendar.duplicated().sum()
)


# --------------------------------------------------
# 5. Data types
# --------------------------------------------------

print("\n========== DATA TYPES ==========")

print(
    calendar.dtypes
)


# --------------------------------------------------
# 6. Public holiday values
# --------------------------------------------------

print("\n========== PUBLIC HOLIDAY ==========")

print(
    calendar["public_holiday"].value_counts(
        dropna=False
    )
)


# --------------------------------------------------
# 7. Event flag values
# --------------------------------------------------

print("\n========== EVENT FLAG ==========")

print(
    calendar["event_flag"].value_counts(
        dropna=False
    )
)


# --------------------------------------------------
# 8. Roadwork flag values
# --------------------------------------------------

print("\n========== ROADWORK FLAG ==========")

print(
    calendar["roadwork_flag"].value_counts(
        dropna=False
    )
)


# --------------------------------------------------
# 9. Date range
# --------------------------------------------------

print("\n========== DATE RANGE ==========")

print(
    "First date:",
    calendar["date"].min()
)

print(
    "Last date:",
    calendar["date"].max()
)