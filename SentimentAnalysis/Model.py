import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re

# Load the dataset
data = pd.read_csv('Twitter_Dataset.csv')

# Preprocess the data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lower case
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
data['Top1'] = data['Top1'].apply(preprocess_text)

# Features and Labels
X = data['Top1']
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Test the model with user input
user_input = input("Enter a sentence to analyze its sentiment: ")
user_input_processed = preprocess_text(user_input)
user_input_vectorized = vectorizer.transform([user_input_processed])
prediction = model.predict(user_input_vectorized)

# Display the result
sentiment = "Positive" if prediction[0] == 1 else "Negative"
print(f'The sentiment of the entered sentence is: {sentiment}')