# Flask Credit Card Approval Prediction API

This is a Flask-based API that provides predictions for credit card approval using a pre-trained XGBoost model. The API accepts demographic and financial data as input, applies necessary preprocessing and feature engineering, and returns predictions for credit card approval status.

## Prerequisites

- **Python Version**: 3.12.7 (Ensure this version is installed)
- **Flask**: Web framework to serve the API
- **XGBoost**: Machine learning library for model loading and predictions
- **Pandas**: For data manipulation and preprocessing
- **Joblib**: For loading the pre-trained scaler

## Setup Instructions

1. **Clone the Repository**:
   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/ziadwaelai/Credit-Card-Approval
   ```

2. **Create a Virtual Environment**:
   It is recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Use the provided `requirements.txt` file to install all necessary dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:
   After setting up the environment, you can run the Flask API server.

   ```bash
   python server.py
   ```

   The API will start and listen on `http://127.0.0.1:5000/`.

## API Endpoint

### `/predict` (POST)

**Description**: This endpoint takes demographic and financial data as JSON input, processes it, and returns predictions for credit card approval (1 for approved, 0 for denied).

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: `application/json`

### Request Body Format

The request body should be a JSON object with the following keys:
- `Num_Children`: Array of integers representing the number of children for each instance.
- `Gender`: Array of strings representing the gender for each instance (`Male` or `Female`).
- `Income`: Array of floats representing the income for each instance.
- `Own_Car`: Array of strings indicating car ownership (`Yes` or `No`).
- `Own_Housing`: Array of strings indicating housing ownership (`Yes` or `No`).

**Example JSON Request**:

```json
{
  "Num_Children": [1, 2],
  "Gender": ["Male", "Female"],
  "Income": [75000, 65000],
  "Own_Car": ["Yes", "No"],
  "Own_Housing": ["Yes", "Yes"]
}
```

### Example Response

The response will include the predictions for each instance in the request data.

**Example JSON Response**:

```json
{
  "predictions": [1, 0]
}
```

### Error Handling

The API includes error handling for common issues:
- If required fields are missing, the API returns a `400 Bad Request` with an appropriate error message.
- Any internal processing errors return a `500 Internal Server Error` with a detailed error message.

**Example Error Response**:

```json
{
  "error": "Missing key in input data: 'Income'"
}
```

## Testing with Postman

1. **Download and Install Postman**:
   If you haven't already, download and install Postman from [here](https://www.postman.com/downloads/).

2. **Create a New Request**:
   - Open Postman and click **New** -> **HTTP Request**.
   - Set the request method to **POST**.
   - Enter the URL: `http://127.0.0.1:5000/predict`.

3. **Add Headers**:
   - In the **Headers** tab, add the following header:
     - `Content-Type`: `application/json`

4. **Add Body**:
   - Select **Body** -> **raw**.
   - Paste your JSON request in the body as shown in the example above.

5. **Send the Request**:
   - Click **Send**.
   - You should see the response with the predictions or an error message.

## Project Components

### `server.py`

- **Model Loading**: Loads a pre-trained XGBoost model (`credit_card_model.json`) and a scaler (`scaler.pkl`) once during server startup.
- **Endpoints**:
  - `/predict`: Accepts JSON data, preprocesses it, applies feature engineering, scales necessary features, and generates predictions.
- **Error Handling**: Provides detailed error messages for missing fields or processing errors.

### `credit_card_model.json`

The trained XGBoost model file that predicts credit card approval based on input features.

### `scaler.pkl`

A pre-trained scaler file, stored as a pickle object, used to scale numerical features to match the distribution of the training data.

### `requirements.txt`

List of required dependencies for running the API.

```plaintext
Flask
pandas
xgboost
joblib
```

## Data Preprocessing and Feature Engineering

The input data undergoes several transformations before being passed to the model:

1. **Categorical Encoding**:
   - `Gender` is mapped to binary values (`Male` = 1, `Female` = 0).
   - `Own_Car` and `Own_Housing` are mapped to binary values (`Yes` = 1, `No` = 0).

2. **Feature Engineering**:
   - `Income_per_Child`: Calculated by dividing `Income` by `Num_Children + 1`.
   - `Financial_Stability`: A binary indicator based on car and housing ownership.
   - Additional features such as `Total_Owned_Assets`, `Gender_Family_Interaction`, `Income_Interaction`, and `Income_Stability_Score`.

3. **Scaling**:
   - Continuous features are scaled using a pre-trained scaler to maintain consistency with the model’s training data.

## Additional Notes

- **Model and Scaler Path**: Ensure that `credit_card_model.json` and `scaler.pkl` are present in the project directory before running the API.
- **Deployment**: This API can be deployed to production on platforms like Heroku or AWS for real-world usage.

--- 
