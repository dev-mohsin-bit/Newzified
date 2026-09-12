import pandas as pd

# Load both datasets
fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

# Add label column: 0 = fake, 1 = real
fake_df['label'] = 0
true_df['label'] = 1

# Combine both into one dataset
df = pd.concat([fake_df, true_df], ignore_index=True)

# Drop subject and date — subject perfectly predicts the label (data leakage), date isn't useful for style-based detection
df = df.drop(columns=['subject', 'date'])

# Shuffle the rows so fake/real aren't in separate blocks
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the result
print("Combined dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nLabel distribution:\n", df['label'].value_counts())
print("\nFirst 3 rows:\n", df.head(3))

# Save this cleaned dataset so we don't have to redo this every time
df.to_csv('dataset/combined_data.csv', index=False)
print("\nSaved combined_data.csv")