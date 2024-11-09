import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to extract features from an audio file
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, duration=30)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Take the mean of the MFCCs
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Load dataset and extract features
def load_dataset(data_path):
    features = []
    labels = []
    for genre in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre)
        if os.path.isdir(genre_path):  # Check if it's a directory
            print(f'Processing genre: {genre}')  # Print the genre being processed
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(genre_path, file)
                    mfccs_mean = extract_features(file_path)
                    features.append(mfccs_mean)
                    labels.append(genre)  # Use the folder name as the label
    return np.array(features), np.array(labels)

# Load dataset
data_path = 'E:\\Machine learning\\MusicGenreClassification\\GZNAT-music-dataset\\Data\\genres_original'  # Update this path
X, y = load_dataset(data_path)

if len(X) == 0:
    print("No audio files found in the specified directory.")
else:
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=2000),  # Increased max_iter
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME')  # Specify the algorithm
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} Accuracy: {accuracy:.2f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name

    print(f'Best Model: {best_model} with accuracy: {best_accuracy:.2f}')

    # Predict the genre of the test file
    test_file_path = 'Test_r.wav'  # Update this path to the location of your test file
    test_features = extract_features(test_file_path).reshape(1, -1)
    test_features = scaler.transform(test_features)  # Scale the test features

    # Use the best model to predict
    if best_model == 'Random Forest':
        model = RandomForestClassifier()
    elif best_model == ' SVC':
        model = SVC()
    elif best_model == 'KNN':
        model = KNeighborsClassifier()
    elif best_model == 'Gradient Boosting':
        model = GradientBoostingClassifier()
    elif best_model == 'Logistic Regression':
        model = LogisticRegression(max_iter=2000)  # Ensure consistency with training
    else:
        model = AdaBoostClassifier(algorithm='SAMME')  # Ensure consistency with training

    model.fit(X_train, y_train)  # Train the best model again
    predicted_genre = model.predict(test_features)
    predicted_genre_label = label_encoder.inverse_transform(predicted_genre)

    print(f'The predicted genre for {test_file_path} is: {predicted_genre_label[0]}')