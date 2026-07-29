import sys
import joblib
import pandas as pd
from extract_features import extract_behavioral_features, get_feature_names

MODEL_PATH = "model.joblib"

def predict_user(prompt_text):
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_PATH} not found. Run train_model.py first.")
        sys.exit(1)
        
    # Extract features from the new prompt
    features = extract_behavioral_features(prompt_text)
    X_new = pd.DataFrame([features], columns=get_feature_names())
    
    # Predict and get probabilities
    prediction = clf.predict(X_new)[0]
    probabilities = clf.predict_proba(X_new)[0]
    classes = clf.classes_
    
    print(f"\nAnalyzing Prompt: '{prompt_text}'")
    print("-" * 40)
    print("Extracted Features:")
    for name, val in zip(get_feature_names(), features):
        print(f"  {name}: {val}")
        
    print("-" * 40)
    print(f"Predicted User Fingerprint: {prediction}")
    print("Confidence breakdown:")
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py \"<your prompt here>\"")
        sys.exit(1)
        
    test_prompt = sys.argv[1]
    predict_user(test_prompt)
