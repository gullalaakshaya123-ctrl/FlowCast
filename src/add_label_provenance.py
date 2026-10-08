import pandas as pd

print("==================================================")
print("     FLOWCAST CONGESTION LABEL PROVENANCE")
print("==================================================")

# Original cleaned traffic data BEFORE missing congestion labels were filled
original = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)

# Traffic data AFTER missing congestion labels were filled
completed = pd.read_csv(
    "data/interim/traffic_cleaned.csv"
)

print("\nOriginal dataset rows:", len(original))
print("Completed dataset rows:", len(completed))

# Check that row counts match
if len(original) != len(completed):
    raise ValueError("Row counts do not match!")

# Identify whether the congestion label was originally available
completed["congestion_label_source"] = (
    original["congestion_level"]
    .notna()
    .map({
        True: "original",
        False: "model_imputed"
    })
)

print("\n========== LABEL PROVENANCE ==========")

print(
    completed["congestion_label_source"]
    .value_counts()
)

print("\n========== LABEL PROPORTIONS ==========")

print(
    completed["congestion_label_source"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Save the dataset
output_path = "data/interim/traffic_cleaned_with_provenance.csv"

completed.to_csv(
    output_path,
    index=False
)

print("\nDataset saved successfully to:")
print(output_path)

print("\n==================================================")
print("     LABEL PROVENANCE COMPLETED")
print("==================================================")