import pandas as pd
import numpy as np
import random

# Define the character replacements
replacements = {
    "B": "В",  # Latin B to Cyrillic В
    "H": "Н",  # Latin H to Cyrillic Н
    "P": "Р",  # Latin P to Cyrillic Р
    "p": "р",  # Latin lowercase p to Cyrillic р
    "c": "с",  # Latin lowercase c to Cyrillic с
    "y": "у",  # Latin lowercase y to Cyrillic у
}

# Function to apply replacements to a string
def replace_chars(text, replacements):
    for orig_char, repl_char in replacements.items():
        text = text.replace(orig_char, repl_char)
    return text

# Load the dataset
df = pd.read_parquet('C:/Users/Alex/Desktop/code/detect_ia/data/better_data/train_essays.parquet')

# Select a random percentage of rows
percentage = 10  # percentage of rows to alter
rows_to_alter = df.sample(frac=percentage/100)

# Apply the character replacements
rows_to_alter['text'] = rows_to_alter['text'].apply(lambda x: replace_chars(x, replacements))

# Append the altered rows back to the original dataframe
df_augmented = pd.concat([df, rows_to_alter])

# Optionally, you can shuffle the dataframe if needed
df_augmented = df_augmented.sample(frac=1).reset_index(drop=True)

# Save the augmented dataframe back to a Parquet file
df_augmented.to_parquet('C:/Users/Alex/Desktop/code/detect_ia/data/better_data/augmented_train_essays.parquet')
