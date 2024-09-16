import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to extract MFCC features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)  # Set a fixed sample rate
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return mfccs.T

# Data preparation
data = []
labels = []

# Replace 'your_dataset_path' with the actual path to your dataset
dataset_path = 'E:/Machine learning/GZNAT-music-dataset/Data/genres_original'

for genre in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']:
    genre_path = f'{dataset_path}/{genre}'
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        features = extract_features(file_path)
        data.append(features)
        labels.append(genre)

# Convert data to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Feature scaling
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))