import pandas as pd
import string

# ✅ Load the dataset correctly before using it
train_df = pd.read_csv("train_reviews.csv")  # Make sure this file exists in the script directory
test_df = pd.read_csv("test_reviews.csv")

# Custom stopword list (since NLTK stopwords may not be available)
custom_stopwords = set(["the", "is", "in", "and", "to", "this", "it", "i", "of", 
                        "for", "on", "with", "a", "an", "at", "by", "so", "but", 
                        "if", "or", "as"])

# Define the preprocessing function
def preprocess_text_simple(text):
    text = str(text).lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenization (splitting words)
    tokens = [word for word in tokens if word not in custom_stopwords]  # Remove stopwords
    return " ".join(tokens)

# ✅ Apply preprocessing after ensuring `train_df` is loaded
train_df["Processed_Review"] = train_df["Review"].apply(preprocess_text_simple)
test_df["Processed_Review"] = test_df["Review"].apply(preprocess_text_simple)

# ✅ Save the preprocessed data
train_df.to_csv("train_reviews_preprocessed.csv", index=False)
test_df.to_csv("test_reviews_preprocessed.csv", index=False)

print("Preprocessing complete. Files saved as 'train_reviews_preprocessed.csv' and 'test_reviews_preprocessed.csv'")
