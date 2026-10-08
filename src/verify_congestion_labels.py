import pandas as pd


# --------------------------------------------------
# 1. Load the cleaned traffic dataset
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_cleaned.csv"
)


# --------------------------------------------------
# 2. Display total records
# --------------------------------------------------

print("========== DATASET CHECK ==========")

print("Total traffic records:", len(traffic))


# --------------------------------------------------
# 3. Check missing congestion labels
# --------------------------------------------------

missing = traffic["congestion_level"].isna().sum()

print(
    "Missing congestion labels:",
    missing
)


# --------------------------------------------------
# 4. Count each congestion level
# --------------------------------------------------

counts = traffic[
    "congestion_level"
].value_counts()


print("\n========== CONGESTION COUNTS ==========")

print(counts)


# --------------------------------------------------
# 5. Calculate percentages
# --------------------------------------------------

percentages = (
    traffic["congestion_level"]
    .value_counts(normalize=True)
    * 100
)


print("\n========== CONGESTION PERCENTAGES ==========")

for level, percentage in percentages.items():

    print(
        f"{level}: {percentage:.2f}%"
    )


# --------------------------------------------------
# 6. Check for unexpected labels
# --------------------------------------------------

expected_labels = {
    "Free-flow",
    "Moderate",
    "Heavy",
    "Severe"
}

actual_labels = set(
    traffic["congestion_level"].dropna().unique()
)

unexpected = actual_labels - expected_labels


print("\n========== LABEL CHECK ==========")

if len(unexpected) == 0:
    print("No unexpected congestion labels found.")
else:
    print(
        "Unexpected labels:",
        unexpected
    )


# --------------------------------------------------
# 7. Final verification
# --------------------------------------------------

if missing == 0 and len(unexpected) == 0:

    print(
        "\nCongestion label verification PASSED."
    )

else:

    print(
        "\nCongestion label verification requires attention."
    )