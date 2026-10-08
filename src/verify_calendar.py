import pandas as pd


# --------------------------------------------------
# 1. Load cleaned calendar dataset
# --------------------------------------------------

calendar = pd.read_csv(
    "data/interim/calendar_cleaned.csv"
)


# --------------------------------------------------
# 2. Basic checks
# --------------------------------------------------

print("========== CALENDAR VERIFICATION ==========")

print("Total records:", len(calendar))

print(
    "Duplicate records:",
    calendar.duplicated().sum()
)


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

print(
    calendar.isnull().sum()
)


# --------------------------------------------------
# 4. Check flag values
# --------------------------------------------------

print("\n========== FLAG VALUES ==========")

print(
    "Public holiday:",
    sorted(calendar["public_holiday"].unique())
)

print(
    "Event flag:",
    sorted(calendar["event_flag"].unique())
)

print(
    "Roadwork flag:",
    sorted(calendar["roadwork_flag"].unique())
)


# --------------------------------------------------
# 5. Date range
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


# --------------------------------------------------
# 6. Final verification
# --------------------------------------------------

if (
    len(calendar) == 151
    and calendar.duplicated().sum() == 0
    and calendar.isnull().sum().sum() == 0
    and set(calendar["public_holiday"].unique()) <= {0, 1}
    and set(calendar["event_flag"].unique()) <= {0, 1}
    and set(calendar["roadwork_flag"].unique()) <= {0, 1}
):

    print("\nCalendar verification PASSED.")

else:

    print("\nCalendar verification requires attention.")