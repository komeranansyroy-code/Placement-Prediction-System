
# --------------------------------------------
# Placement Prediction Dataset Preprocessing
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------
# Read Original Dataset
# --------------------------------------------

input_file = "D:/Pycham projects/Placement_predict/Dataset/placement_predict_50K_Raw.csv"

output_file = "D:/Pycham projects/Placement_predict/Dataset/final_preprocess_M2.csv"

df = pd.read_csv(input_file)


# --------------------------------------------
# Create a copy
# Original dataset remains unchanged
# --------------------------------------------

processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)


# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------

processed_df = processed_df.drop_duplicates().copy()


# --------------------------------------------
# Handle Missing Values
# --------------------------------------------

# Numeric Columns
numeric_cols = processed_df.select_dtypes(
    include=["number"]
).columns.tolist()


# Fill numerical missing values with Median
for col in numeric_cols:

    if processed_df[col].isnull().any():

        median_value = processed_df[col].median()

        processed_df[col] = processed_df[col].fillna(
            median_value
        )


# Categorical Columns
categorical_cols = processed_df.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# Fill categorical missing values with Mode
for col in categorical_cols:

    if processed_df[col].isnull().any():

        mode_values = processed_df[col].mode()

        if len(mode_values) > 0:

            processed_df[col] = processed_df[col].fillna(
                mode_values.iloc[0]
            )


# --------------------------------------------
# Clean Text Data
# --------------------------------------------

for col in categorical_cols:

    # Remove leading and trailing spaces
    processed_df[col] = processed_df[col].str.strip()

    # Convert text to lowercase
    processed_df[col] = processed_df[col].str.lower()


# --------------------------------------------
# Label Encoding
# --------------------------------------------

for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = encoder.fit_transform(
        processed_df[col].astype(str)
    )


# --------------------------------------------
# Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

processed_df[numeric_cols] = scaler.fit_transform(
    processed_df[numeric_cols]
)


# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------

processed_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------
# Display Results
# --------------------------------------------

print("\nPreprocessing Completed Successfully!")

print("Original Dataset Shape :", df.shape)

print("Processed Dataset Shape:", processed_df.shape)

print("Saved File :", output_file)


# --------------------------------------------
# Verify Missing Values
# --------------------------------------------

print("\nRemaining Missing Values:")

print(processed_df.isnull().sum())


# --------------------------------------------
# Verify Original Dataset Was Not Modified
# --------------------------------------------

df_check = pd.read_csv(input_file)

if df.equals(df_check):

    print("\nOriginal dataset was NOT modified.")

else:

    print("\nWARNING: Original dataset was modified!")


# --------------------------------------------
# Completion Message
# --------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

