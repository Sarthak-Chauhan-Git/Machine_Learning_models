import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('e:\\Machine learning\\StockMarketPrediction\\Combined_News_DJIA.csv')

# Print the columns to debug
print("Columns in DataFrame:", df.columns)

# Check if 'Label' column exists
if 'Label' in df.columns:
    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])

    # Combine the news headlines into a single feature
    df['Combined_News'] = df[['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 
                               'Top6', 'Top7', 'Top8', 'Top9', 'Top10', 
                               'Top11', 'Top12', 'Top13', 'Top14', 'Top15', 
                               'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 
                               'Top21', 'Top22', 'Top23', 'Top24', 'Top25']].fillna('').astype(str).agg(' '.join, axis=1)

    # Feature Engineering: Convert text data to numerical data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Combined_News'])  # Features
    y = df['Label']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Make predictions for the entire dataset
    all_predictions = model.predict(X)

    # Assuming the dataset has a 'Company' column or you can extract company names from headlines
    # Here, we will create a dummy company list based on the index of the DataFrame.
    # If your dataset has specific companies, replace this with the actual company names.
    companies = [f'Company {i+1}' for i in range(len(df))]  # Example: Company 1, Company 2, ...

    # Create a DataFrame to store results
    results = pd.DataFrame({
        'Company': companies,
        'Predicted_Profitability': all_predictions
    })

    # Filter companies predicted to be profitable
    profitable_companies = results[results['Predicted_Profitability'] == 1]

    # Output the names of companies predicted to be profitable
    print("Predicted Profitable Companies:")
    print(profitable_companies)

else:
    print("Error: 'Label' column not found in the DataFrame. Available columns are:")
    print(df.columns)