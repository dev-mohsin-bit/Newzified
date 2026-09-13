import pandas as pd

train_df = pd.read_csv('dataset/ag_news/train.csv')
test_df = pd.read_csv('dataset/ag_news/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:", train_df.columns.tolist())

print("\nClass distribution (train):\n", train_df['Class Index'].value_counts())

print("\nFirst 3 rows:\n", train_df.head(3))
# Keep only Sports (2) and Sci/Tech (4) for style diversity
diverse_df = train_df[train_df['Class Index'].isin([2, 4])].copy()

# Combine Title + Description into one 'text' column (matching your original structure)
diverse_df['text'] = diverse_df['Title'] + ' ' + diverse_df['Description']

# Add label = 1 (real)
diverse_df['label'] = 1

# Keep only the columns we need
diverse_df = diverse_df[['text', 'label']]

print("\nNew diverse real-news samples:", diverse_df.shape)
print(diverse_df.head(3))

# Save this for the next step (merging)
diverse_df.to_csv('dataset/ag_news_diverse.csv', index=False)
print("\nSaved dataset/ag_news_diverse.csv")