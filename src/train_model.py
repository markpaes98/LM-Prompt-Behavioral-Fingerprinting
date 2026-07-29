import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from extract_features import extract_behavioral_features, get_feature_names

DATA_PATH = "data/sample_prompts.jsonl"
MODEL_PATH = "model.joblib"

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def main():
    print("Load data:")
    df = load_data(DATA_PATH)
    
    print("Extracting behavioral features:")
    # Apply feature extraction to the prompt column
    features = df['prompt'].apply(extract_behavioral_features)
    
    # Convert list of features into a DataFrame
    X = pd.DataFrame(features.tolist(), columns=get_feature_names())
    y = df['user_id']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training Random Forest Classifier:")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nModel Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
