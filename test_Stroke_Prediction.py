from Stroke_Prediction import Predict
import pandas as pd
import joblib
Model1 = joblib.load('Model1.pkl')




def test_Predict():
    Patient_A = ['Male', 67.0, 0, 1, 'Yes', 'Private', 'Urban', 228.69, 36.600000, 'formerly smoked']
    assert Predict(Patient_A)=='It is highly possible that the patient is stroke'

    Patient_B = ['Female', 39.0, 0, 0, 'Yes', 'Self-employed', 'Rural', 89.86, 24.4, 'never smoked']
    assert Predict(Patient_B)=='It is highly possible that the patient is not stroke'

    Patient_C = ['Male', 81.0, 1, 1, 'Yes', 'Private', 'Urban', 250.89, 28.1, 'smokes']
    assert Predict(Patient_C)=='It is highly possible that the patient is stroke'

    Patient_D = ['Female', 71.0, 0, 0, 'Yes', 'Self-employed', 'Urban', 195.71, 34.1, 'formerly smoked']
    assert Predict(Patient_D)=='It is highly possible that the patient is stroke'

    Patient_E = ['Female', 19.0, 0, 0, 'No', 'Private', 'Urban', 76.57, 26.6, 'Unknown']
    assert Predict(Patient_E)=='It is highly possible that the patient is not stroke'

    Patient_F = ['Female', 51.0, 0, 0, 'Yes', 'Govt_job', 'Rural', 92.95, 23.9, 'never smoked']
    assert Predict(Patient_E)=='It is highly possible that the patient is not stroke'

    Patient_G = ['Female', 31.0, 0, 0, 'Yes', 'Self-employed', 'Urban', 206.59, 41.4, 'smokes']
    assert Predict(Patient_G)=='It is highly possible that the patient is not stroke'


