import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ProcessPoolExecutor
import warnings
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to extract features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        
        features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Data preparation
def process_file(args):
    file_path, genre = args
    features = extract_features(file_path)
    return features, genre

def main():
    data = []
    labels = []

    dataset_path = 'E:/Machine learning/GZNAT-music-dataset/Data/genres_original'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        file_genre_pairs = [
            (os.path.join(dataset_path, genre, file), genre)
            for genre in genres
            for file in os.listdir(os.path.join(dataset_path, genre))
            if file.endswith(('.wav', '.mp3'))
        ]
        
        results = list(executor.map(process_file, file_genre_pairs))

    for features, genre in results:
        if features is not None:
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

    # Create and train the Random Forest classifier with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()