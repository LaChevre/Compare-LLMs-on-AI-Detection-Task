from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_length_boxplot(csv_filenames):
    lengths = []
    for filename in csv_filenames:
        df = pd.read_csv(filename)
        lengths.append([len(str(x)) for x in df['text']])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(lengths, labels=csv_filenames)
    plt.ylabel('Length of Strings')
    plt.title('Distribution of String Lengths in Datasets')
    plt.grid(True)
    plt.show()


def check_exact_duplicates(csv_filenames):
    # Load the datasets from CSV filenames
    df1 = pd.read_csv(csv_filenames[0])
    df2 = pd.read_csv(csv_filenames[1])
    
    # Find duplicates between df1 and df2
    duplicates = pd.merge(df1, df2, on="text", how="inner")
    
    # Print the number of duplicates and total elements
    print(f"Number of duplicates between {csv_filenames[0]} and {csv_filenames[1]}: {len(duplicates)}")
    print(f"Number of elements in {csv_filenames[0]}: {len(df1)}")
    print(f"Number of elements in {csv_filenames[1]}: {len(df2)}")


def check_near_duplicates(csv_filenames, threshold=0.8):
    # Load datasets
    df1 = pd.read_csv(csv_filenames[0])
    df2 = pd.read_csv(csv_filenames[1])

    # Vectorize texts
    vectorizer = TfidfVectorizer()
    tfidf_df1 = vectorizer.fit_transform(df1['text'])
    tfidf_df2 = vectorizer.transform(df2['text'])

    # Compute cosine similarity between datasets
    cosine_sim = cosine_similarity(tfidf_df1, tfidf_df2)
    near_duplicates = cosine_sim > threshold

    # Count and print the number of near-duplicate pairs
    count_near_duplicates = sum(sum(near_duplicates))
    print(f"Found {count_near_duplicates} near-duplicate pairs between datasets with threshold {threshold}.")


def check_unique_near_duplicates(csv_filenames, threshold=0.9):
    df1 = pd.read_csv(csv_filenames[0])
    df2 = pd.read_csv(csv_filenames[1])
    
    vectorizer = TfidfVectorizer()
    tfidf_df1 = vectorizer.fit_transform(df1['text'])
    tfidf_df2 = vectorizer.transform(df2['text'])
    
    cosine_sim = cosine_similarity(tfidf_df1, tfidf_df2)
    
    matched_df1 = set()
    matched_df2 = set()
    for i, row in enumerate(cosine_sim):
        for j, sim in enumerate(row):
            if sim > threshold and i not in matched_df1 and j not in matched_df2:
                matched_df1.add(i)
                matched_df2.add(j)
                
    print(f"Unique near-duplicate pairs found: {len(matched_df1)}")


csv_filenames = ['data/valid.csv', 'data/train.csv', 'data/out_validation_all_one.csv', 'data/out_validation_all_zero.csv', 'data/out_validation_missleading.csv']

check_unique_near_duplicates(csv_filenames)
# check_near_duplicates(csv_filenames)
check_exact_duplicates(csv_filenames)
plot_mean_length_boxplot(csv_filenames)
