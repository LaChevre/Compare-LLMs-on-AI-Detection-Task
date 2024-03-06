import pandas as pd
import random

# Function to modify the text
# This is a placeholder function - you can define your own logic here
def replace_with_cyrillic(text):
    # Dictionary mapping Latin characters to their Cyrillic counterparts
    replacements = {
        "B": "В",  # Latin B to Cyrillic В
        "H": "Н",  # Latin H to Cyrillic Н
        "P": "Р",  # Latin P to Cyrillic Р
        "p": "р",  # Latin lowercase p to Cyrillic р
        "c": "с",  # Latin lowercase c to Cyrillic с
        "y": "у",  # Latin lowercase y to Cyrillic у
    }

    # Replace each character in the text
    modified_text = ''.join(replacements.get(char, char) for char in text)
    
    return modified_text

import random

def randomly_add_spaces_and_newlines(text):
    # Function to randomly insert an extra character after existing characters (space or newline)
    def random_insert(original, char):
        parts = original.split(char)
        modified = char.join(part + (char if random.choice([True, False]) else '') for part in parts)
        return modified

    # Randomly add spaces after spaces and newlines after newlines
    text_with_extra_spaces = random_insert(text, ' ')
    text_with_extra_spaces_and_newlines = random_insert(text_with_extra_spaces, '\n')

    # Add 20 newlines and spaces at the end
    text_with_extra_spaces_and_newlines += ' \n' * 20

    return text_with_extra_spaces_and_newlines

import random

def introduce_typos(text):
    # list of common typo mistakes to introduce
    typo_options = {
        'a': ['s', 'q', 'w', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g'],
        'g': ['f', 't', 'y', 'h'],
        'h': ['g', 'y', 'u', 'j'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k'],
        'k': ['j', 'i', 'o', 'l'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k', 'l'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'k', 'l'],
        'p': ['o', 'l'],
        'q': ['w', 'a', 's'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'h', 'j'],
        'v': ['c', 'b', 'f', 'g'],
        'w': ['q', 'e', 'a', 's'],
        'x': ['z', 'c', 'd', 's'],
        'y': ['t', 'u', 'g', 'h'],
        'z': ['x', 'a', 's']
        # Add more if needed
    }

    # Function to make a typo
    def make_typo(char):
        if char.lower() in typo_options:
            return random.choice(typo_options[char.lower()])
        else:
            return char

    # Introduce typos
    typo_text = ''.join(make_typo(char) if random.randint(1, 30) == 1 else char for char in text)

    return typo_text


def modify_csv_and_add_columns(df):
    # Load the CSV file
    

    # Check if the required columns exist
    if 'text' in df.columns and 'generated' in df.columns:
        # Apply the replace_with_cyrillic function
        df['cyrillic_text'] = df.apply(lambda row: replace_with_cyrillic(row['text']) if row['generated'] == 1 else row['text'], axis=1)

        # Apply the introduce_typos function
        df['text_with_typos'] = df.apply(lambda row: introduce_typos(row['text']) if row['generated'] == 1 else row['text'], axis=1)

        # Apply the randomly_add_spaces_and_newlines function
        df['text_with_extra_spaces_and_newlines'] = df.apply(lambda row: randomly_add_spaces_and_newlines(row['text']) if row['generated'] == 1 else row['text'], axis=1)
    else:
        print("Error: Required columns not found in the CSV file.")
        return

    # Save the modified DataFrame back to a new CSV file
    df.to_csv('modified_file.csv', index=False)

    return df


# Example usage
csv_file = 'C:/Users/Alex/Desktop/code/detect_ia/data/out_validation_all_one.csv'  # Replace with your own CSV file
# Save the filtered DataFrame to a new CSV file
output_csv_file = 'C:/Users/Alex/Desktop/code/detect_ia/data/out_validation_missleading.csv'  # Specify your desired output file path


df = pd.read_csv(csv_file)

total_rows = len(df)
print(f"Total rows: {total_rows}")

# Number of rows where 'generated' is 0
rows_with_generated_0 = len(df[df['generated'] == 0])
print(f"Rows where 'generated' is 0: {rows_with_generated_0}")

# Number of rows where 'generated' is 1
rows_with_generated_1 = len(df[df['generated'] == 1])
print(f"Rows where 'generated' is 1: {rows_with_generated_1}")



modified_df = modify_csv_and_add_columns(df)
#print(modified_df.head(50))

# Filter the DataFrame to include only rows where 'generated' is 1
filtered_df = modified_df[modified_df['generated'] == 1]

filtered_df.to_csv(output_csv_file, index=False)  # Set 'index=False' to not include the DataFrame index in the CSV file

print(f"Filtered DataFrame saved to '{output_csv_file}'")

# Print the first 50 rows of the filtered DataFrame
print(filtered_df.head(50))
