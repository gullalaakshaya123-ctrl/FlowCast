import pandas as pd


# --------------------------------------------------
# 1. Load raw calendar dataset
# --------------------------------------------------

calendar = pd.read_csv(
    "data/raw/calendar_events.csv"
)


# --------------------------------------------------
# 2. Clean holiday name
# --------------------------------------------------

calendar.loc[
    calendar["public_holiday"] == 0,
    "holiday_name"
] = "No Holiday"


# --------------------------------------------------
# 3. Clean event name
# --------------------------------------------------

calendar.loc[
    calendar["event_flag"] == 0,
    "event_name"
] = "No Event"


# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("========== AFTER CLEANING ==========")

print(
    calendar.isnull().sum()
)


# --------------------------------------------------
# 5. Display holiday information
# --------------------------------------------------

print("\n========== HOLIDAY INFORMATION ==========")

print(
    calendar[
        ["public_holiday", "holiday_name"]
    ].value_counts()
)


# --------------------------------------------------
# 6. Display event information
# --------------------------------------------------

print("\n========== EVENT INFORMATION ==========")

print(
    calendar[
        ["event_flag", "event_name"]
    ].value_counts()
)


# --------------------------------------------------
# 7. Save cleaned calendar dataset
# --------------------------------------------------

calendar.to_csv(
    "data/interim/calendar_cleaned.csv",
    index=False
)


print(
    "\nCalendar dataset cleaned successfully."
)

print(
    "Cleaned calendar dataset saved successfully."
)