import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def extract_features(y=None, file_path=None, max_length=1000):
    if file_path is not None:
        y, sr = librosa.load(file_path)
    elif y is not None:
        sr = 22050  # Default sampling rate for librosa
    else:
        raise ValueError("Either 'y' or 'file_path' must be provided.")

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # Ensure all features have the same length
    mfcc = librosa.util.fix_length(mfcc, size=max_length, axis=1)
    chroma = librosa.util.fix_length(chroma, size=max_length, axis=1)
    mel = librosa.util.fix_length(mel, size=max_length, axis=1)
    spectral_contrast = librosa.util.fix_length(spectral_contrast, size=max_length, axis=1)
    zero_crossing_rate = librosa.util.fix_length(zero_crossing_rate, size=max_length, axis=1)

    # Flatten 2D arrays to 1D
    mfcc_flat = mfcc.flatten()
    chroma_flat = chroma.flatten()
    mel_flat = mel.flatten()
    spectral_contrast_flat = spectral_contrast.flatten()
    zero_crossing_rate_flat = zero_crossing_rate.flatten()
    
    # Combine all features
    features = np.concatenate([mfcc_flat, chroma_flat, mel_flat, spectral_contrast_flat, zero_crossing_rate_flat])
    
    return features

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

# Define path to your dataset
dataset_path = "E:/Machine learning/GZNAT-music-dataset/Data/genres_original"  # Replace with your actual dataset path

# List of genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

X = []
y = []

# Process files for each genre
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    print(f"Processing {genre} files...")
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path=file_path)
            if features is not None:
                X.append(features)
                y.append(genre)
                
                # Augment data by adding noise
                y_augmented = add_noise(librosa.load(file_path)[0])
                augmented_features = extract_features(y=y_augmented)
                if augmented_features is not None:
                    X.append(augmented_features)
                    y.append(genre)
                    
    print(f"Finished processing {genre} files.")

if len(X) == 0:
    print("No files were successfully processed. Please check your file paths and audio files.")
    exit()

X = np.array(X)
y = np.array(y)

print(f"Total samples processed: {len(X)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the SVM model with the best parameters
model = SVC(**best_params)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=genres))