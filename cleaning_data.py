import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
df_essays = pd.read_csv('C:/Users/Alex/Desktop/code/detect_ia/data/train_essays.csv')
df_drcat = pd.read_csv('C:/Users/Alex/Desktop/code/detect_ia/data/train_drcat_01.csv')
df_training = pd.read_csv('C:/Users/Alex/Desktop/code/detect_ia/data/Training_Essay_Data.csv')

# Rename 'label' to 'generated' in df_drcat
df_drcat.rename(columns={'label': 'generated'}, inplace=True)

# Select only the 'text' and 'generated' columns
df_essays = df_essays[['text', 'generated']]
df_drcat = df_drcat[['text', 'generated']]
df_training = df_training[['text', 'generated']]  # Assuming this structure

# Concatenate all DataFrames
combined_df = pd.concat([df_essays, df_drcat, df_training], ignore_index=True)

# Stratified split to maintain the same proportion of 'generated' in train, test, and validation sets
train_df, temp_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['generated'], random_state=42)
test_df, valid_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['generated'], random_state=42)

# Save the datasets to new CSV files
train_df.to_csv('C:/Users/Alex/Desktop/code/detect_ia/data/train.csv', index=False)
test_df.to_csv('C:/Users/Alex/Desktop/code/detect_ia/data/test.csv', index=False)
valid_df.to_csv('C:/Users/Alex/Desktop/code/detect_ia/data/valid.csv', index=False)

print(f"Train CSV with {len(train_df)} rows saved as 'train.csv'")
print(f"Test CSV with {len(test_df)} rows saved as 'test.csv'")
print(f"Valid CSV with {len(valid_df)} rows saved as 'valid.csv'")
