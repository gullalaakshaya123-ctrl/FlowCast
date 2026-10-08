import pandas as pd

print("==================================================")
print("   FLOWCAST ORIGINAL-LABEL EVALUATION DATASET")
print("==================================================")

# Load dataset with label provenance
df = pd.read_csv(
    "data/interim/traffic_cleaned_with_provenance.csv"
)

print("\nTotal records:", len(df))

# Keep only originally observed congestion labels
original_labels = df[
    df["congestion_label_source"] == "original"
].copy()

print("\nOriginal-label records:", len(original_labels))

# Check that no model-imputed labels remain
remaining_imputed = (
    original_labels["congestion_label_source"]
    == "model_imputed"
).sum()

print("Model-imputed records remaining:", remaining_imputed)

# Check congestion classes
print("\n========== CONGESTION LABEL DISTRIBUTION ==========")

print(
    original_labels["congestion_level"]
    .value_counts()
)

print("\n========== CONGESTION LABEL PERCENTAGE ==========")

print(
    original_labels["congestion_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Save dataset
output_path = (
    "data/processed/original_label_dataset.csv"
)

original_labels.to_csv(
    output_path,
    index=False
)

print("\nDataset saved successfully to:")
print(output_path)

print("\n==================================================")
print("   ORIGINAL-LABEL DATASET CREATED")
print("==================================================")