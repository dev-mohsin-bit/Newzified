import pandas as pd

# Load your original combined dataset
original_df = pd.read_csv('dataset/combined_data.csv')
print("Original combined dataset:", original_df.shape)

# Load the new diverse real-news data
diverse_df = pd.read_csv('dataset/ag_news_diverse.csv')
print("Diverse real-news data:", diverse_df.shape)

# Take a sample of 10,000 from the diverse set, so we don't overwhelm the original balance
diverse_sample = diverse_df.sample(n=10000, random_state=42)

# Combine title-less structure: original_df has 'title','text','label' -- match structure
# We'll just use 'text' and 'label' from both for consistency
original_simple = original_df[['text', 'label']]

# Merge original + new diverse sample
final_df = pd.concat([original_simple, diverse_sample], ignore_index=True)

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal merged dataset shape:", final_df.shape)
print("\nLabel distribution:\n", final_df['label'].value_counts())

# Save
final_df.to_csv('dataset/combined_data_v2.csv', index=False)
print("\nSaved dataset/combined_data_v2.csv")