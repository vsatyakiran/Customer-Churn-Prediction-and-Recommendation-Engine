'''
  check API.md to know how to use the API to get the prediction
'''

from flask import Flask, request, jsonify
import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model
model = pickle.load(open('rf.pkl', 'rb'))
scale = pickle.load(open('scaler.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
gender_encode = {'Male':1, 'Female':0}
churn = {'Yes':1, 'No':0}

total_features_taken = ['Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceUsage1',
       'ServiceUsage2', 'ServiceUsage3', 'AvgMonthlySpend',
       'TotalServiceUsage', 'PaymentMethod']

@app.route('/predict', methods=['POST'])
def home():
    # Get the data from the POST request.
    data = request.get_json()
    print(data)
    # Make prediction using model loaded from disk as per the data.
    gender = data['gender']
    monthly_charges = data['monthly_charges']
    total_charges = data['total_charges']
    service_usage1 = data['service_usage1']
    service_usage2 = data['service_usage2']
    service_usage3 = data['service_usage3']
    tenure = data['tenure']
    payment_method = data['payment_method']

    avg_monthly_spend = total_charges/tenure
    total_service = service_usage1 + service_usage2 + service_usage3

    # encoding gender
    gender = gender_encode[gender]
    # encoding payment method
    payment_method = ohe.transform(np.array(payment_method).reshape(1, -1))[0]
    # scaling
    scalable_data = np.array([[tenure, monthly_charges, total_charges, service_usage1,service_usage2, service_usage3, total_service, avg_monthly_spend]])
    scalable_data = scale.transform(scalable_data)[0]
    tenure = scalable_data[0]
    monthly_charges = scalable_data[1]
    total_charges = scalable_data[2]
    service_usage1 = scalable_data[3]
    service_usage2 = scalable_data[4]
    service_usage3 = scalable_data[5]
    total_service = scalable_data[6]
    avg_monthly_spend = scalable_data[7]

    # Final data for prediction
    data = [gender, tenure, monthly_charges, total_charges, service_usage1, service_usage2, service_usage3, avg_monthly_spend, total_service, payment_method[0], payment_method[1], payment_method[2]] 

    prediction = model.predict_proba([data])
    churn_prob = prediction[0][1]
    
    return jsonify({'churn_probability(%)': round(churn_prob*100, 2)})
    

if __name__ == '__main__':
    app.run(debug=True)
