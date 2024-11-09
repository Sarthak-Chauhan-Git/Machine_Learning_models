import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Use TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Experiment with different models
models = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier()
}

best_model_name = None
best_model = None
best_accuracy = 0

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model  # Save the model object

# Display classification report for the best model
print(f'\nBest Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%')
print(classification_report(y_test, best_model.predict(X_test_vectorized)))

# Test the best model with user input
user_input = input("Enter a sentence to analyze its sentiment: ")
user_input_processed = preprocess_text(user_input)
user_input_vectorized = vectorizer.transform([user_input_processed])
prediction = best_model.predict(user_input_vectorized)

# Display the result
sentiment = "Positive" if prediction[0] == 1 else "Negative"
print(f'The sentiment of the entered sentence is: {sentiment}')