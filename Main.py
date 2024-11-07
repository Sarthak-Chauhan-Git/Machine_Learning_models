import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def extract_features(file_path, max_length=1000):
    try:
        y, sr = librosa.load(file_path)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Ensure all features have the same length
        mfcc = librosa.util.fix_length(mfcc, size=max_length, axis=1)
        chroma = librosa.util.fix_length(chroma, size=max_length, axis=1)
        mel = librosa.util.fix_length(mel, size=max_length, axis=1)
        
        # Flatten 2D arrays to 1D
        mfcc_flat = mfcc.flatten()
        chroma_flat = chroma.flatten()
        mel_flat = mel.flatten()
        
        # Combine all features
        features = np.concatenate([mfcc_flat, chroma_flat, mel_flat])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

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
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
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

# Train the model
print("Training the model...")
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
print("Making predictions...")
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict the genre of a new audio file
def predict_genre(file_path, model, scaler):
    # Extract features from the new audio file
    features = extract_features(file_path)
    
    if features is not None:
        # Scale the features
        features_scaled = scaler.transform([features])  # Scale the features
        # Make prediction
        prediction = model.predict(features_scaled)
        return prediction[0]  # Return the predicted genre
    else:
        return None

# Loop to ask for predictions until the user decides to quit
while True:
    user_input = input("Do you want to predict the genre of a new audio file? (yes/no): ").strip().lower()
    
    if user_input == 'yes':
        new_audio_file = input("Please provide the path to the audio file: ").strip()
        predicted_genre = predict_genre(new_audio_file, svm, scaler)

        if predicted_genre:
            print(f"The predicted genre for the audio file is: {predicted_genre}")
        else:
            print("Failed to predict the genre. Please check the audio file.")
    
    elif user_input == 'no':
        print("Exiting the prediction loop.")
        break
    
    elif user_input == 'quit':
        print("Quitting the program.")
        break
    
    else:
        print(" Invalid input. Please respond with 'yes' to predict, 'no' to exit, or 'quit' to quit the program.") 
    