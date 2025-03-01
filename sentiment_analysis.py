import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed training and testing datasets
train_df = pd.read_csv("train_reviews_preprocessed.csv")
test_df = pd.read_csv("test_reviews_preprocessed.csv")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000)

# Convert text data to numerical features
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["Processed_Review"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["Processed_Review"])

y_train = train_df["Liked"]  # Labels (0 = Negative, 1 = Positive, 2 = Neutral)
y_test = test_df["Liked"]

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_tfidf, y_train)

# Test the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
print("Model and Vectorizer saved successfully!")

# Function to predict sentiment
def predict_sentiment(review):
    processed_review = [review]  # Wrap input in a list
    transformed_review = tfidf_vectorizer.transform(processed_review)  # Transform using TF-IDF
    prediction = model.predict(transformed_review)[0]  # Get prediction
    sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    return sentiment_map[prediction]

# Example usage
sample_review = input("Enter a review: ")
print("Predicted Sentiment:", predict_sentiment(sample_review))
