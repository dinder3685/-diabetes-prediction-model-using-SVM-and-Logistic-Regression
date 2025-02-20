# Diabetes Prediction using SVM and Logistic Regression

## Overview
This project is a machine learning-based diabetes prediction system using the **PIMA Indians Diabetes Dataset**. It employs **Support Vector Machines (SVM)** and **Logistic Regression** to classify whether a person has diabetes based on medical attributes.

## Dataset
The dataset contains multiple medical predictor variables (features) and one target variable:
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- **Target:** 1 (Diabetic) or 0 (Non-Diabetic).

## Requirements
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
### Dependencies:
- `pandas`
- `numpy`
- `sklearn`
- `pickle`

## Model Training
1. Load the dataset into a Pandas DataFrame.
2. Preprocess the data by handling missing values and scaling features using `StandardScaler`.
3. Split the data into training and testing sets using `train_test_split`.
4. Train the model using **SVM** and **Logistic Regression**.
5. Evaluate the model using accuracy scores and classification reports.

## Usage
### Training the Model
Run the following script to train the model:
```bash
python train.py
```
### Predicting Diabetes
To predict whether a person has diabetes:
```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def is_diabetic(features):
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

# Example usage
features = [2, 120, 70, 30, 80, 25.0, 0.5, 45]  # Sample input
print(is_diabetic(features))
```

## Saving & Loading the Model
To save the trained model:
```python
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
```
To load and use the model:
```python
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
```

## Future Improvements
- Implement a web-based UI using Flask or FastAPI.
- Enhance model performance with feature engineering and hyperparameter tuning.
- Integrate deep learning models like Neural Networks.

## License
This project is open-source and available under the MIT License.

## Author
Ahmed

