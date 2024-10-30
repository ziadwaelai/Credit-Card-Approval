from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model once when the app starts
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model('credit_card_model.json')
        return model
    except Exception as e:
        print("Error loading model:", e)
        raise Exception("Model not found or could not be loaded. Check the model file path.")

def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except Exception as e:
        print("Error loading scaler:", e)
        raise Exception("Scaler not found or could not be loaded. Check the scaler file path.")

model = load_model()  # Load the model once and reuse it
scaler = load_scaler()  # Load the scaler once and reuse it

MODEL_FEATURE_ORDER = [
    'Num_Children', 'Income', 'Income_per_Child', 'Total_Owned_Assets', 
    'Income_Interaction', 'Income_Stability_Score', 'Gender', 'Own_Housing', 
    'Own_Car', 'Financial_Stability', 'Large_Family', 'Gender_Family_Interaction'
]
binary_list = ['Gender', 'Own_Housing', 'Own_Car', 'Financial_Stability', 'Large_Family', 'Gender_Family_Interaction']

def data_preprocessing_and_feature_engineering(data):
    try:
        # Mapping categorical variables to numerical values
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
        data['Own_Car'] = data['Own_Car'].map({'Yes': 1, 'No': 0})
        data['Own_Housing'] = data['Own_Housing'].map({'Yes': 1, 'No': 0})
        
        # Feature engineering
        data['Income_per_Child'] = data['Income'] / (data['Num_Children'] + 1)  # Avoid division by zero
        data['Financial_Stability'] = ((data['Own_Car'] == 1) & (data['Own_Housing'] == 1)).astype(int)
        data['Large_Family'] = (data['Num_Children'] > 3).astype(int)
        data['Total_Owned_Assets'] = data['Own_Car'] + data['Own_Housing']
        data['Gender_Family_Interaction'] = data['Gender'] * data['Num_Children']
        data['Income_Interaction'] = data['Income'] * data['Num_Children']
        data['Income_Stability_Score'] = data['Income'] * (data['Own_Housing'] + data['Own_Car'])

        # Separate the binary and continuous columns
        data_to_scale = data.drop(columns=binary_list)
        data_scaled_values = scaler.transform(data_to_scale)  # Use transform instead of fit_transform
        data_scaled = pd.DataFrame(data_scaled_values, columns=data_to_scale.columns)
        
        # Add binary features back to the scaled DataFrame
        for col in binary_list:
            data_scaled[col] = data[col].values
        
        # Ensure correct column order
        data = data_scaled[MODEL_FEATURE_ORDER]
        return data

    except Exception as e:
        print("Error in data preprocessing and feature engineering:", e)
        raise Exception("Data preprocessing and feature engineering failed. Check the input data format.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data_json = request.get_json()

        # Convert JSON data to a DataFrame
        data = pd.DataFrame({
            "Num_Children": data_json["Num_Children"],
            "Gender": data_json["Gender"],
            "Income": data_json["Income"],
            "Own_Car": data_json["Own_Car"],
            "Own_Housing": data_json["Own_Housing"]
        })
        
        # Apply data preprocessing and feature engineering
        data = data_preprocessing_and_feature_engineering(data)
        print(data)
        # Generate predictions
        predictions = model.predict(data).tolist()

        # Return predictions as JSON
        return jsonify({"predictions": predictions})

    except KeyError as e:
        print(f"Missing key in input data: {e}")
        return jsonify({"error": f"Missing key in input data: {e}"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)